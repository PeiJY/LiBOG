from model.critic import Critic
from utils.logger import log_to_tb_train
from utils.utils import clip_grad_norms
from .rollout import rollout
from utils.utils import set_seed
from dataset.generate_dataset import sample_batch_task_cec21,get_train_set_cec21,sample_batch_task_cec21_lifelong, get_train_set_cec21_lifelong
from dataset.cec_test_func import *
import numpy as np
from .task import TaskForTrain
from env import SubprocVectorEnv,DummyVectorEnv

from pbo_env import L2E_env,MadDE,sep_CMA_ES,PSO,DE
from expr.tokenizer import MyTokenizer
import torch
from tqdm import tqdm
from utils.utils import torch_load_cpu, get_inner_model, get_surrogate_gbest
import os
import psutil
class Data_Memory():
    def __init__(self) -> None:
        self.teacher_cost=[]
        self.stu_cost=[]
        self.baseline_cost=[]
        self.gap=[]
        self.baseline_gap=[]
        self.expr=[]
    
    def clear(self):
        del self.teacher_cost[:]
        del self.stu_cost[:]
        del self.baseline_cost[:]
        del self.gap[:]
        del self.baseline_gap[:]
        del self.expr[:]

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.gap_rewards=[]
        self.b_rewards=[]

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.gap_rewards[:]
        del self.b_rewards[:]
        
    def clone(self):
        new_memory = Memory()
        new_memory.actions = [copy.deepcopy(t) for t in self.actions]
        new_memory.states = [t.clone() for t in self.states]
        new_memory.logprobs = [t.clone() for t in self.logprobs]
        new_memory.rewards = [t.clone() for t in self.rewards]
        new_memory.gap_rewards = [t.clone() for t in self.gap_rewards]
        new_memory.b_rewards = [t.clone() for t in self.b_rewards]
        return new_memory
    
    def extend(self, another_memory):
        self.actions.extend([copy.deepcopy(t) for t in another_memory.actions])
        self.states.extend([t.clone() for t in another_memory.states])
        self.logprobs.extend([t.clone() for t in another_memory.logprobs])
        self.rewards.extend([t.clone() for t in another_memory.rewards])
        self.gap_rewards.extend([t.clone() for t in another_memory.gap_rewards])
        self.b_rewards.extend([t.clone() for t in another_memory.b_rewards])

class trainer(object):
    def __init__(self,model,opts) -> None:
        self.actor=model
        self.critic=Critic(opts)
        self.opts=opts
        self.optimizer=torch.optim.Adam([{'params':self.actor.parameters(),'lr':opts.lr}] + 
                                        [{'params':self.critic.parameters(),'lr':opts.lr_critic}])
        # figure out the lr schedule
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)
        if opts.use_cuda:
            # move to cuda
            self.actor.to(opts.device)
            self.critic.to(opts.device)
        # load model to cuda
        # move optimizer's data onto chosen device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    def set_training(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        self.critic.train()
    
    def set_evaling(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        self.critic.eval()

    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic=get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    # def save(self, epoch):
    #     print('Saving model and state...')
    #     torch.save(
    #         {
    #             'actor': get_inner_model(self.actor).state_dict(),
    #             'critic':get_inner_model(self.critic).state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #             'rng_state': torch.get_rng_state(),
    #             'cuda_rng_state': torch.cuda.get_rng_state_all(),
    #         },
    #         os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
    #     )
    def save_lifelong(self, epoch,task_id):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic':get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}-{}.pt'.format(epoch,task_id))
        )

    # inference for training
    def start_training(self,tb_logger):
        opts=self.opts
        process = psutil.Process(os.getpid())
        # parallel vector
        self.vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
        
        # construct the dataset
        set_seed(42)
        train_set,train_pro_id=get_train_set_cec21_lifelong(self.opts)
        
        # construct parallel environment
        # learning_env
        learning_env_list=[lambda e=train_set[0]: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        learning_env=self.vector_env(learning_env_list)
        # teacher_env
        if self.opts.teacher=='madde':
            if self.opts.tea_step == 'step':
                madde_maxfes = round((opts.max_fes / opts.population_size) * (4 + 2 * opts.dim * opts.dim) / 2)
            else:
                madde_maxfes = opts.max_fes
            teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): MadDE(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=madde_maxfes) for i in range(opts.batch_size)]
            
        elif self.opts.teacher=='cmaes':
            teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): sep_CMA_ES(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,sigma=opts.cmaes_sigma) for i in range(opts.batch_size)]
        elif self.opts.teacher=='pso':
            teacher_env_list=[lambda e=None: PSO(ps=opts.population_size,dim=opts.dim,max_fes=opts.max_fes,min_x=opts.min_x,max_x=opts.max_x,pho=0.2) for i in range(opts.batch_size)]
        elif self.opts.teacher=='de':
            teacher_env_list=[lambda e=None: DE(dim=opts.dim,ps=opts.population_size,min_x=opts.min_x,max_x=opts.max_x,max_fes=opts.max_fes) for i in range(opts.batch_size)]
        else:
            assert True, f'The selecting {self.opts.teacher} teacher is currently not supported!!'
        teacher_env=self.vector_env(teacher_env_list)
        # random_env (for comparison)
        random_env_list=[lambda e=train_set[0]: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        random_env=self.vector_env(random_env_list)
        
        # get surrogate gbest and init cost
        self.surrogate_gbest=get_surrogate_gbest(teacher_env,train_set,train_pro_id,opts.batch_size,seed=999,fes=opts.max_fes)
        
        task=TaskForTrain(learning_env,teacher_env,random_env,opts.batch_size,opts)
        tokenizer=MyTokenizer()

        # for epoch in range(100):
        #     test_ratio=rollout(opts,self,epoch,tb_logger,tokenizer,testing=True)
        update_step=0

        test_ratio_list=[]

        epoch_len=18
        best_test_ratio = float('inf')
        best_avg_gap = 0
        best_gap_epoch = 0
        # begin training 
        for epoch in range(opts.epoch_start,opts.epoch_end):
            self.lr_scheduler.step(epoch)
            
            self.set_training()
            # logging
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with RNN lr={:.3e} for run {} on task {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name, opts.current_task_id) , flush=True)
            # start training
            epoch_step=epoch_len * (opts.max_fes // opts.population_size // opts.skip_step // opts.n_step) * opts.k_epoch
            pbar = tqdm(total = epoch_step, desc = 'training',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            

            total_gap=0
            training_instance_ids = []
            for b in range(epoch_len):
                batch_step,bat_gap,data_memory,instance_ids=self.train_batch(task,update_step,tokenizer,pbar,epoch,tb_logger)
                training_instance_ids.extend(instance_ids)
                data_memory.clear()

                update_step+=batch_step
                total_gap+=bat_gap

            
            avg_gap=total_gap/epoch_len
            pbar.close()
            
            # save model
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                                       epoch == opts.epoch_end - 1): self.save_lifelong(epoch,opts.current_task_id)
            
            # rollout
            if epoch%2==0 or epoch==opts.epoch_end-1:
                test_ratio=rollout(opts,self,epoch,tb_logger,tokenizer)
                test_ratio_list.append(test_ratio)
                if epoch == opts.epoch_start:
                    best_test_ratio=test_ratio
                else:
                    if best_test_ratio<test_ratio:
                        best_test_ratio=test_ratio
                        best_test_epoch=epoch
                        
            if epoch == opts.epoch_start:
                best_test_epoch=opts.epoch_start
                best_avg_gap=avg_gap
                best_gap_epoch=opts.epoch_start
            else:
                if best_avg_gap>avg_gap:
                    best_avg_gap=avg_gap
                    best_gap_epoch=epoch
            
            
            training_instance_ids = list(set(training_instance_ids))
            # todo: delete
            # log to screen
            # print(f'test_ratio_list:{test_ratio_list}')
            print(f'best_test_epoch:{best_test_epoch}')
            print(f'best_test_ratio:{best_test_ratio}')
            print(f'current_avg_gap:{avg_gap}')
            print(f'best_avg_gap:{best_avg_gap}')
            print(f'best_gap_epoch:{best_gap_epoch}')
            print(f"training functions: {training_instance_ids}")
        # close the parallel vector_env
        learning_env.close()
        teacher_env.close()
        random_env.close()


    # training for one batch 
    def train_batch(self,task:TaskForTrain,pre_step,tokenizer,pbar,epoch,tb_logger=None):
        
        max_step=self.opts.max_fes//(self.opts.population_size*self.opts.skip_step)
        data_memory=Data_Memory()
        memory=Memory()

        # reset
        set_seed()
        
        # sample task for training
        instances,ids=sample_batch_task_cec21_lifelong(self.opts)

        tea_pop,stu_population=task.reset(instances)

        baseline_pop=copy.deepcopy(stu_population)

        pop_feature=task.state(stu_population)
        pop_feature=torch.FloatTensor(pop_feature).to(self.opts.device)

        # record the pre_population
        pre_stu_pop=stu_population
        pre_baseline_pop=stu_population

        # record infomation
        data_memory.teacher_cost.append([p.gbest_cost for p in tea_pop])
        data_memory.stu_cost.append([p.gbest_cost for p in stu_population])
        data_memory.baseline_cost.append([p.gbest_cost for p in stu_population])

        gamma = self.opts.gamma
        eps_clip = self.opts.eps_clip
        n_step = self.opts.n_step
        k_epoch=self.opts.k_epoch

        t=0
        total_gap=0
        # for logging
        current_step=pre_step
        
        # record init cost
        init_cost=[p.gworst_cost for p in tea_pop]

        # bat_step=self.opts.max_fes//self.opts.population_size//self.opts.skip_step
        is_done=False
        while not is_done:
            t_s=t

            bl_val_detached = []
            bl_val = []

            while t-t_s < n_step:
                # get feature
                memory.states.append(pop_feature)

                # using model to generate expr
                if self.opts.require_baseline:
                    seq,const_seq,log_prob,rand_seq,rand_c_seq,action_dict=self.actor(pop_feature,save_data=True)
                else:
                    seq,const_seq,log_prob,action_dict=self.actor(pop_feature,save_data=True)
                
                
                # next_pop为使用expr更新的population，target_pop为teacher
                target_pop,next_pop,baseline_pop,expr,is_done=task.step(stu_population,self.opts.skip_step,seq,const_seq,tokenizer,rand_seq,rand_c_seq,baseline_pop)
                
                # get reward
                total_reward,gap_reward,base_reward,gap=task.reward(learning_population=next_pop,target_population=target_pop,reward_method=self.opts.reward_func,
                                                                    base_reward_method=self.opts.b_reward_func,max_step=max_step,epoch=epoch,s_init_cost=init_cost,
                                                                    s_gbest=self.surrogate_gbest,pre_learning_population=pre_stu_pop,ids=ids)
                gap_reward=torch.FloatTensor(gap_reward).to(self.opts.device)
                base_reward=torch.FloatTensor(base_reward).to(self.opts.device)
                total_reward=torch.FloatTensor(total_reward).to(self.opts.device)

                if torch.any(torch.isnan(total_reward)):
                    print(f'gap_reward:{gap_reward},base_reward:{base_reward},total_reward:{total_reward}')
                    assert True, 'nan in reward!!'

                memory.gap_rewards.append(gap_reward)
                memory.b_rewards.append(base_reward)

                
                total_gap+=np.mean(gap)

                # critic network
                baseline_val_detached,baseline_val=self.critic(pop_feature)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # store reward for ppo
                memory.actions.append(action_dict)
                memory.logprobs.append(log_prob)
                memory.rewards.append(total_reward)

                # store data
                data_memory.teacher_cost.append([p.gbest_cost for p in target_pop])
                data_memory.stu_cost.append([p.gbest_cost for p in next_pop])
                data_memory.gap.append(gap)
                data_memory.baseline_cost.append([p.gbest_cost for p in baseline_pop])
                data_memory.expr.append(expr)

                # total_baseline_reward,baseline_reward,baseline_base_reward,baseline_gap=task.reward(baseline_pop,target_pop,self.opts.reward_func,self.opts.b_reward_func,max_step,s_init_cost,s_gbest)
                total_baseline_reward,baseline_reward,baseline_base_reward,baseline_gap=task.reward(learning_population=baseline_pop,target_population=target_pop,reward_method=self.opts.reward_func,
                                                                    base_reward_method=self.opts.b_reward_func,max_step=max_step,epoch=epoch,s_init_cost=init_cost,
                                                                    s_gbest=self.surrogate_gbest,pre_learning_population=pre_baseline_pop,ids=ids)
                

                # next step 
                stu_population=next_pop
                pre_stu_pop=next_pop
                pre_baseline_pop=baseline_pop

                t=t+1

                # next state
                pop_feature = task.state(stu_population)
                pop_feature=torch.FloatTensor(pop_feature).to(self.opts.device)

                
                if is_done:
                    # update surrogate gbest
                    for i,id in enumerate(ids):
                        min_cost=min(data_memory.teacher_cost[-1][i],data_memory.stu_cost[-1][i])
                        self.surrogate_gbest[id]=min(self.surrogate_gbest[id],min_cost)
                    break

            t_time=t-t_s
            

            # begin updating network in PPO style
            old_actions = memory.actions
            old_states = torch.stack(memory.states).detach() 
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            old_value = None
            for _k in range(k_epoch):
                
                if _k == 0:
                    logprobs = memory.logprobs
                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):

                        # get new action_prob
                        log_p = self.actor(old_states[tt],fix_action = old_actions[tt])

                        logprobs.append(log_p)
                        
                        baseline_val_detached, baseline_val = self.critic(old_states[tt])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)
                logprobs = torch.stack(logprobs).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
               
                R = self.critic(pop_feature)[0]
                critic_output=R.clone()
                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()
                # calculate loss
                loss = baseline_loss + reinforce_loss
                
                # see if loss is nan
                if torch.isnan(loss):
                    print(f'baseline_loss:{baseline_loss}')
                    print(f'reinforce_loss:{reinforce_loss}')
                    assert True, 'nan found in loss!!'

                # update gradient step
                self.set_training()
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradient norm and get (clipped) gradient norms for logging if needed
                grad_norms = clip_grad_norms(self.optimizer.param_groups)[0]

                # perform gradient descent
                self.optimizer.step()

                current_step+=1
                

                # logging to tensorboard
                if (not self.opts.no_tb) and (tb_logger is not None):
                    if current_step % self.opts.log_step == 0:
                        log_to_tb_train(tb_logger,self,Reward,grad_norms, memory.rewards,memory.gap_rewards,memory.b_rewards,gap,reinforce_loss,baseline_loss,log_prob,current_step)
                pbar.update(1)
            memory.clear_memory()

        
        # return batch step
        return current_step-pre_step,total_gap/t,data_memory,ids






class trainer_LiBOG(object):
    def __init__(self,model,opts) -> None:
        self.ewc_lambda = opts.ewc_lambda
        self.task_id = 0
        self.actor=model
        self.old_parameters = []
        self.bsf_actor_model = None
        self.bsf_actor_model_performance = None
        # self.last_batch_memory = None
        # self.last_old_value = None
        self.save_memory_last_epoch = False
        self.final_epoch_memory = None
        self.last_pop_feature = None
        self.fisher_infos = []
        self.critic=Critic(opts)
        self.opts=opts
        self.optimizer=torch.optim.Adam([{'params':self.actor.parameters(),'lr':opts.lr}] + 
                                        [{'params':self.critic.parameters(),'lr':opts.lr_critic}])
        # figure out the lr schedule
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        if opts.use_cuda:
            # move to cuda
            self.actor.to(opts.device)
            self.critic.to(opts.device)
        # load model to cuda
        # move optimizer's data onto chosen device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    def set_training(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        self.critic.train()
    
    def set_evaling(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        self.critic.eval()

    def next_task(self):
        self.fisher_infos.append(self.compute_fisher_information())
        self.old_parameters.append({name: param.clone() for name, param in self.actor.named_parameters()})
        self.task_id += 1
        self.bsf_actor_model_performance = None
        self.bsf_actor_model = None
        self.final_epoch_memory = None
        self.last_pop_feature  =None

    def ewc_penalty(self): 
        total_penalty = 0.0
        for task_id in range(len(self.old_parameters)):
            old_param = self.old_parameters[task_id]
            fisher_info = self.fisher_infos[task_id]
            penalty = 0.0
            for name, param in self.actor.named_parameters():
                if name in old_param:
                    penalty += (fisher_info[name] * (param - old_param[name]) ** 2).sum()
            total_penalty += penalty
        return self.ewc_lambda * total_penalty

    def compute_fisher_information(self):
        torch.autograd.set_grad_enabled(True)
        # if self.last_batch_memory is None:
        if self.final_epoch_memory is None:
            raise ValueError('the memory of last batch is None, no data can be used for compute fisher information')
        
        # self.actor.eval()
        # self.critic.eval()
        # memory = self.last_batch_memory
        memory = self.final_epoch_memory
        old_actions = memory.actions
        # old_states = torch.stack(memory.states).detach() 
        old_states = torch.stack(memory.states)
        # old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        # Evaluating old actions and values :
        logprobs = []
        bl_val_detached = []
        bl_val = []
        gamma = self.opts.gamma
        eps_clip = self.opts.eps_clip
        for tt in range(len(old_states)):

            # get new action_prob
            log_p = self.actor(old_states[tt],fix_action = old_actions[tt])

            logprobs.append(log_p)
            
            baseline_val_detached, baseline_val = self.critic(old_states[tt])

            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)
        logprobs = torch.stack(logprobs).view(-1)
        bl_val_detached = torch.stack(bl_val_detached).view(-1)
        bl_val = torch.stack(bl_val).view(-1)

        # get traget value for critic
        Reward = []
        reward_reversed = memory.rewards[::-1]
        
        pop_feature=torch.FloatTensor(self.last_pop_feature).to(self.opts.device)
        R = self.critic(pop_feature)[0]
        critic_output=R.clone()
        for r in range(len(reward_reversed)):
            R = R * gamma + reward_reversed[r]
            Reward.append(R)
        # clip the target:
        Reward = torch.stack(Reward[::-1], 0)
        Reward = Reward.view(-1)

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs.detach())

        # Finding Surrogate Loss:
        advantages = Reward - bl_val_detached

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
        reinforce_loss = -torch.min(surr1, surr2).mean()
        # old_value = self.last_old_value
        # define baseline loss
        # if old_value is None:
        #     baseline_loss = ((bl_val - Reward) ** 2).mean()
        #     old_value = bl_val.detach()
        # else:
        #     vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
        #     v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
        #     baseline_loss = v_max.mean()
        # calculate loss
        # loss = baseline_loss + reinforce_loss
        
        fisher = {name: torch.zeros_like(param) for name, param in self.actor.named_parameters()}

        # self.actor.train()
        self.set_training()
        self.actor.zero_grad()
        # loss.backward()
        reinforce_loss.backward()
        
        for name, param in self.actor.named_parameters():
            fisher[name] += param.grad.pow(2).detach()
        return fisher

    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        
        self.old_parameters = load_data.get('old_parameters',{}),
        # self.last_batch_memory = load_data.get('last_batch_memory',None)
        self.self.final_epoch_memory = load_data.get('self.final_epoch_memory',None)
        # self.last_old_value = load_data.get('last_old_value',{})
        self.last_pop_feature = load_data.get('last_pop_feature',{})
        self.fisher_infos = load_data.get('fisher_infos',{})
        self.bsf_actor_model_performance  = load_data.get('bsf_actor_model_performance',None)
        self.bsf_actor_model = load_data.get('bsf_actor_model',None)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic=get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    # def save(self, epoch):
    #     print('Saving model and state...')
    #     torch.save(
    #         {
    #             'actor': get_inner_model(self.actor).state_dict(),
    #             'critic':get_inner_model(self.critic).state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #             'rng_state': torch.get_rng_state(),
    #             'cuda_rng_state': torch.cuda.get_rng_state_all(),
    #         },
    #         os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
    #     )
    def save_lifelong(self, epoch, task_id):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic':get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'old_parameters':self.old_parameters,
                # 'last_batch_memory':self.last_batch_memory.clone(),
                'final_epoch_memory':None if self.final_epoch_memory is None else self.final_epoch_memory.clone(),
                # 'last_old_value':self.last_old_value,
                'last_pop_feature':self.last_pop_feature,
                'fisher_infos':self.fisher_infos,
                'bsf_actor_model_performance':self.bsf_actor_model_performance,
            },
            os.path.join(self.opts.save_dir, 'epoch-{}-{}.pt'.format(epoch,task_id))
        )

    # inference for training
    def start_training(self,tb_logger):
        self.save_memory_last_epoch =False
        opts=self.opts
        process = psutil.Process(os.getpid())
        # parallel vector
        self.vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
        
        # construct the dataset
        set_seed(42)
        train_set,train_pro_id=get_train_set_cec21_lifelong(self.opts)
        
        # construct parallel environment
        # learning_env
        learning_env_list=[lambda e=train_set[0]: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        learning_env=self.vector_env(learning_env_list)
        # teacher_env
        if self.opts.teacher=='madde':
            if self.opts.tea_step == 'step':
                madde_maxfes = round((opts.max_fes / opts.population_size) * (4 + 2 * opts.dim * opts.dim) / 2)
            else:
                madde_maxfes = opts.max_fes
            teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): MadDE(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=madde_maxfes) for i in range(opts.batch_size)]
            
        elif self.opts.teacher=='cmaes':
            teacher_env_list=[lambda e=copy.deepcopy(train_set[0]): sep_CMA_ES(dim=opts.dim,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,sigma=opts.cmaes_sigma) for i in range(opts.batch_size)]
        elif self.opts.teacher=='pso':
            teacher_env_list=[lambda e=None: PSO(ps=opts.population_size,dim=opts.dim,max_fes=opts.max_fes,min_x=opts.min_x,max_x=opts.max_x,pho=0.2) for i in range(opts.batch_size)]
        elif self.opts.teacher=='de':
            teacher_env_list=[lambda e=None: DE(dim=opts.dim,ps=opts.population_size,min_x=opts.min_x,max_x=opts.max_x,max_fes=opts.max_fes) for i in range(opts.batch_size)]
        else:
            assert True, f'The selecting {self.opts.teacher} teacher is currently not supported!!'
        teacher_env=self.vector_env(teacher_env_list)
        # random_env (for comparison)
        random_env_list=[lambda e=train_set[0]: L2E_env(dim=opts.dim,ps=opts.population_size,problem=e,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method=opts.boarder_method) for i in range(opts.batch_size)]
        random_env=self.vector_env(random_env_list)
        
        # get surrogate gbest and init cost
        self.surrogate_gbest=get_surrogate_gbest(teacher_env,train_set,train_pro_id,opts.batch_size,seed=999,fes=opts.max_fes)
        
        task=TaskForTrain(learning_env,teacher_env,random_env,opts.batch_size,opts)
        tokenizer=MyTokenizer()

        # for epoch in range(100):
        #     test_ratio=rollout(opts,self,epoch,tb_logger,tokenizer,testing=True)
        update_step=0

        test_ratio_list=[]

        epoch_len=18
        best_test_ratio = float('inf')
        best_avg_gap = 0
        best_gap_epoch = 0
        self.set_training()
        # begin training 
        for epoch in range(opts.epoch_start,opts.epoch_end):
            if epoch == opts.epoch_end-1:
                self.save_memory_last_epoch =True
                self.final_epoch_memory = Memory()
            self.lr_scheduler.step(epoch)
            

            # logging
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with RNN lr={:.3e} for run {} on task {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name, opts.current_task_id) , flush=True)
            # start training
            epoch_step=epoch_len * (opts.max_fes // opts.population_size // opts.skip_step // opts.n_step) * opts.k_epoch
            pbar = tqdm(total = epoch_step, desc = 'training',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            

            total_gap=0
            training_instance_ids = []
            for b in range(epoch_len):

                batch_step,bat_gap,data_memory,instance_ids=self.train_batch(task,update_step,tokenizer,pbar,epoch,tb_logger)
                training_instance_ids.extend(instance_ids)
                data_memory.clear()

                update_step+=batch_step
                total_gap+=bat_gap

            
            avg_gap=total_gap/epoch_len
            pbar.close()
            
            # save model
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                                       epoch == opts.epoch_end - 1): self.save_lifelong(epoch,opts.current_task_id)
            
            # rollout
            if epoch%2==0  or epoch==opts.epoch_end-1:
                test_ratio=rollout(opts,self,epoch,tb_logger,tokenizer)
                test_ratio_list.append(test_ratio)
                if epoch == opts.epoch_start:
                    best_test_ratio=best_test_ratio
                else:
                    if best_test_ratio<test_ratio:
                        best_test_ratio=test_ratio
                        best_test_epoch=epoch
                        
            if epoch == opts.epoch_start:
                best_test_epoch=opts.epoch_start
                best_avg_gap=avg_gap
                best_gap_epoch=opts.epoch_start
                self.bsf_actor_model_performance = best_avg_gap
                self.bsf_actor_model = copy.deepcopy(self.actor)
                # self.bsf_actor_model.load_state_dict(model.state_dict())
            else:
                if best_avg_gap>avg_gap:
                    best_avg_gap=avg_gap
                    best_gap_epoch=epoch
                    self.bsf_actor_model_performance = best_avg_gap
                    self.bsf_actor_model =copy.deepcopy(self.actor)
            
            
            training_instance_ids = list(set(training_instance_ids))
            # todo: delete
            # log to screen
            # print(f'test_ratio_list:{test_ratio_list}')
            print(f'best_test_epoch:{best_test_epoch}')
            print(f'best_test_ratio:{best_test_ratio}')
            print(f'current_avg_gap:{avg_gap}')
            print(f'best_avg_gap:{best_avg_gap}')
            print(f'best_gap_epoch:{best_gap_epoch}')
            print(f"training functions: {training_instance_ids}")
        

        # close the parallel vector_env
        learning_env.close()
        teacher_env.close()
        random_env.close()




    # training for one batch 
    def train_batch(self,task:TaskForTrain,pre_step,tokenizer,pbar,epoch,tb_logger=None):
        self.set_training()
        max_step=self.opts.max_fes//(self.opts.population_size*self.opts.skip_step)
        data_memory=Data_Memory()
        memory=Memory()

        # reset
        set_seed()
        
        # sample task for training
        instances,ids=sample_batch_task_cec21_lifelong(self.opts)

        tea_pop,stu_population=task.reset(instances)

        baseline_pop=copy.deepcopy(stu_population)

        pop_feature=task.state(stu_population)
        pop_feature=torch.FloatTensor(pop_feature).to(self.opts.device)

        # record the pre_population
        pre_stu_pop=stu_population
        pre_baseline_pop=stu_population

        # record infomation
        data_memory.teacher_cost.append([p.gbest_cost for p in tea_pop])
        data_memory.stu_cost.append([p.gbest_cost for p in stu_population])
        data_memory.baseline_cost.append([p.gbest_cost for p in stu_population])

        gamma = self.opts.gamma
        eps_clip = self.opts.eps_clip
        n_step = self.opts.n_step
        k_epoch=self.opts.k_epoch

        t=0
        total_gap=0
        # for logging
        current_step=pre_step
        
        # record init cost
        init_cost=[p.gworst_cost for p in tea_pop]

        # bat_step=self.opts.max_fes//self.opts.population_size//self.opts.skip_step
        is_done=False
        while not is_done:
            t_s=t

            bl_val_detached = []
            bl_val = []

            while t-t_s < n_step:
                # get feature
                memory.states.append(pop_feature)
                self.set_training()
                # using model to generate expr
                if self.opts.require_baseline:
                    seq,const_seq,log_prob,rand_seq,rand_c_seq,action_dict=self.actor(pop_feature,save_data=True)
                else:
                    seq,const_seq,log_prob,action_dict=self.actor(pop_feature,save_data=True)
                
                
                # next_pop为使用expr更新的population，target_pop为teacher
                target_pop,next_pop,baseline_pop,expr,is_done=task.step(stu_population,self.opts.skip_step,seq,const_seq,tokenizer,rand_seq,rand_c_seq,baseline_pop)
                
                # get reward
                total_reward,gap_reward,base_reward,gap=task.reward(learning_population=next_pop,target_population=target_pop,reward_method=self.opts.reward_func,
                                                                    base_reward_method=self.opts.b_reward_func,max_step=max_step,epoch=epoch,s_init_cost=init_cost,
                                                                    s_gbest=self.surrogate_gbest,pre_learning_population=pre_stu_pop,ids=ids)
                gap_reward=torch.FloatTensor(gap_reward).to(self.opts.device)
                base_reward=torch.FloatTensor(base_reward).to(self.opts.device)
                total_reward=torch.FloatTensor(total_reward).to(self.opts.device)

                if torch.any(torch.isnan(total_reward)):
                    print(f'gap_reward:{gap_reward},base_reward:{base_reward},total_reward:{total_reward}')
                    assert True, 'nan in reward!!'

                memory.gap_rewards.append(gap_reward)
                memory.b_rewards.append(base_reward)

                
                total_gap+=np.mean(gap)

                # critic network
                baseline_val_detached,baseline_val=self.critic(pop_feature)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # store reward for ppo
                memory.actions.append(action_dict)
                memory.logprobs.append(log_prob)
                
                memory.rewards.append(total_reward)

                # store data
                data_memory.teacher_cost.append([p.gbest_cost for p in target_pop])
                data_memory.stu_cost.append([p.gbest_cost for p in next_pop])
                data_memory.gap.append(gap)
                data_memory.baseline_cost.append([p.gbest_cost for p in baseline_pop])
                data_memory.expr.append(expr)

                # total_baseline_reward,baseline_reward,baseline_base_reward,baseline_gap=task.reward(baseline_pop,target_pop,self.opts.reward_func,self.opts.b_reward_func,max_step,s_init_cost,s_gbest)
                total_baseline_reward,baseline_reward,baseline_base_reward,baseline_gap=task.reward(learning_population=baseline_pop,target_population=target_pop,reward_method=self.opts.reward_func,
                                                                    base_reward_method=self.opts.b_reward_func,max_step=max_step,epoch=epoch,s_init_cost=init_cost,
                                                                    s_gbest=self.surrogate_gbest,pre_learning_population=pre_baseline_pop,ids=ids)
                

                # next step 
                stu_population=next_pop
                pre_stu_pop=next_pop
                pre_baseline_pop=baseline_pop

                t=t+1

                # next state
                pop_feature = task.state(stu_population)
                self.last_pop_feature = copy.deepcopy(pop_feature)
                pop_feature=torch.FloatTensor(pop_feature).to(self.opts.device)

                
                if is_done:
                    # update surrogate gbest
                    for i,id in enumerate(ids):
                        min_cost=min(data_memory.teacher_cost[-1][i],data_memory.stu_cost[-1][i])
                        self.surrogate_gbest[id]=min(self.surrogate_gbest[id],min_cost)
                    break

            t_time=10
            

            # begin updating network in PPO style
            old_actions = memory.actions
            old_states = torch.stack(memory.states).detach() 
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            old_value = None
            # self.last_batch_memory = memory.clone()
            for _k in range(k_epoch):
                
                if _k == 0:
                    logprobs = memory.logprobs
                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):

                        # get new action_prob
                        log_p = self.actor(old_states[tt],fix_action = old_actions[tt])

                        logprobs.append(log_p)
                        
                        baseline_val_detached, baseline_val = self.critic(old_states[tt])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)
                logprobs = torch.stack(logprobs).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
               
                R = self.critic(pop_feature)[0]
                
                critic_output=R.clone()
                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                    # self.last_old_value = old_value.clone()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()
                # calculate loss
                loss = baseline_loss + reinforce_loss
                ewc_loss = self.ewc_penalty() # TODO check if loss is positive or negtive, then decide ewc penalty
                loss += ewc_loss
                
                imitation_loss = self.compute_imitation_loss(old_states, old_actions)
                loss += imitation_loss
                
                # print(baseline_loss,reinforce_loss,ewc_loss,imitation_loss)
                # see if loss is nan
                if torch.isnan(loss):
                    print(f'baseline_loss:{baseline_loss}')
                    print(f'reinforce_loss:{reinforce_loss}')
                    assert True, 'nan found in loss!!'

                # update gradient step
                self.actor.train()
                for name, module in self.actor.named_modules():
                    if not module.training:
                        print(f"! Warning: {name} is in eval mode!")
                        module.train()
                
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradient norm and get (clipped) gradient norms for logging if needed
                grad_norms = clip_grad_norms(self.optimizer.param_groups)[0]

                # perform gradient descent
                self.optimizer.step()

                current_step+=1
                

                # logging to tensorboard
                if (not self.opts.no_tb) and (tb_logger is not None):
                    if current_step % self.opts.log_step == 0:
                        log_to_tb_train(tb_logger,self,Reward,grad_norms, memory.rewards,memory.gap_rewards,memory.b_rewards,gap,reinforce_loss,baseline_loss,log_prob,current_step)
                pbar.update(1)
            if self.save_memory_last_epoch:
                self.final_epoch_memory.extend(memory)
            memory.clear_memory()

        
        # return batch step
        return current_step-pre_step,total_gap/t,data_memory,ids


    def compute_imitation_loss(self, states, action):

        if self.bsf_actor_model is None:
            return 0.0
        self.set_training()
        imitation_loss = 0
        for tt in range(len(states)):
            s = states[tt].clone()
            s.requires_grad_()
            # a = {key:[t.clone().detach().requires_grad_() for t in tensor_list] for key, tensor_list in action[tt].items()}
            # a = {key:[t.clone() for t in tensor_list] for key, tensor_list in action[tt].items()}
            a = action[tt]
            
            s_prob = self.actor(s,fix_action = a) + 1e-8
            s_prob.requires_grad_()
            # student_probs.append(s_prob)
            t_prob = self.bsf_actor_model(s,fix_action = a) + 1e-8
            t_prob.requires_grad_()
            # teacher_probs.append(t_prob)
            kl_div = torch.nn.functional.kl_div(torch.log(s_prob / s_prob.sum(dim=-1, keepdim=True)), t_prob / t_prob.sum(dim=-1, keepdim=True), reduction='batchmean')
            # imitation_loss += max(kl_div, 0)
            imitation_loss += torch.maximum(kl_div,torch.tensor(0.0,device=kl_div.device))
        
        imitation_loss /= len(states)
        return self.opts.intask_consolidation_lambda * imitation_loss
    
  