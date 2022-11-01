import random
import time
from dataclasses import asdict

import numpy as np
import torch as T
from fastprogress.fastprogress import progress_bar

from common.logger import MetricLogger

debug = False


def run(CONFIG, env, agent_type, hyper_params, run_num=None, save_model=False, verbose=False):
    num_states = CONFIG['env']['num_states']
    num_actions = CONFIG['env']['num_actions']
    exp_decay_explor = CONFIG['agent']['exp_decay_explor']
    discounted_return = CONFIG['experiment'].get('discounted_return')
    num_runs = CONFIG['experiment']['num_runs'] if run_num is None else 1
    num_episodes = CONFIG['experiment']['num_episodes']
    gamma = CONFIG['env']['gamma']
    timeout = CONFIG['experiment']['timeout']  # math.inf
    log_window = CONFIG['experiment']['log_window']
    num_steps = CONFIG['experiment']['num_steps']
    rep_param = None if CONFIG['agent']['rep_type'] is None else CONFIG['agent']['rep_param']
    AGENT_REG = CONFIG['agent_register']
    AGENT_DEFAULT_PARAMS = CONFIG['agent_default_params']
    dest_dir = CONFIG['experiment']['dest_dir']
    experiment_name = agent_type + '_' + '_'.join([f'{k}_{v}' for k, v in hyper_params.items()])
    env_name = CONFIG['env']['name']

    start = time.time()
    print(env_name, experiment_name)

    for run in progress_bar(range(num_runs)):
        seed = run_num = run if run_num is None else run_num
        metric_logger = MetricLogger(experiment_name, run_num, dest_dir)
        metric_logger.add_meta('hyper_params', hyper_params)
        run_start = time.time()
        # seed everything
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        T.cuda.manual_seed_all(seed)

        env.reset()
        env.seed(seed)
        # only need to set for continuous action space
        # env.action_space.seed(seed)

        # -----> gamma seed
        if exp_decay_explor:
            epsilon = 1
        else:
            epsilon = 0

        agent = AGENT_REG[agent_type]()
        agent_info = {
            "num_actions": num_actions,
            "num_states": num_states,
            "update_interval": CONFIG['agent']['update_interval'],
            "seq_len": CONFIG['agent']['seq_len'],
            # default total planning step to seq_len
            "total_planning": hyper_params['seq_len'],
            "discount": gamma,
            "epsilon": epsilon,
            "seed": run,
            "rep_type": CONFIG['agent']['rep_type'],
            "rep_param": rep_param,
            "net_type": CONFIG['agent'].get('net_type'),
            "opt": CONFIG['agent'].get('opt'),
        }
        agent_info.update(asdict(AGENT_DEFAULT_PARAMS[agent_type]))
        agent_info.update(hyper_params)
        agent.agent_init(agent_info)

        total_return = 0
        total_step_count = 0
        total_ep_count = 0
        restart = True

        while True:
            if restart:
                print(f"episode {total_ep_count}",end='\r')
                episode_start = time.time()
                restart = False
                ep_return = 0
                ep_step_count = 0
                if discounted_return:
                    cum_discount = 1

                obs = env.reset()
                obs = T.from_numpy(obs).unsqueeze(dim=0)
                action = agent.agent_start(obs)

            obs, reward, is_terminal, _ = env.step(action)
            obs = T.from_numpy(obs).unsqueeze(dim=0)

            if debug:
                env.render(mode='human')
            print(total_step_count, end='\r')
            if discounted_return:
                ep_return += cum_discount * reward
                cum_discount *= gamma
            else:
                ep_return += reward
            # metric_logger.add_scalar('episodic_returns', ep_return)
            total_return += reward
            ep_step_count += 1
            total_step_count += 1
            state = obs

            if exp_decay_explor:
                agent.epsilon *= CONFIG['agent']['exp_decay_rate']
            if ep_step_count == timeout or is_terminal:
                if ep_step_count == timeout:
                    agent.agent_end(reward, state, append_buffer=False)
                elif is_terminal:
                    agent.agent_end(reward, state, append_buffer=True)

                restart = True
                episode_end = time.time()
                total_ep_count += 1

                if total_step_count >= num_steps:
                    metric_logger.add_list('episodic_returns', [round(ep_return, 4)]*(ep_step_count-total_step_count+num_steps))
                    break
                else:
                    metric_logger.add_list('episodic_returns', [round(ep_return, 4)]*ep_step_count)

            else:
                action = agent.agent_step(reward, state)

            if total_step_count % log_window == 0:
                metric_logger.add_scalar('log_returns', total_return)
                if verbose:
                    print(f"Step {total_step_count}, Episode {total_ep_count}, Episodic Return {ep_return:.4f}, Total Return {total_return:.4f}")
        metric_logger.dump()
        if save_model:
            (dest_dir/"models").mkdir(parents=True, exist_ok=True)
            T.save(agent.nn.state_dict(), dest_dir/f'models/{experiment_name}_{run_num}.torch')

    end = time.time()

    print("total run time: ", end - start)

    return experiment_name, np.mean(metric_logger.metrics["episodic_returns"][experiment_name]), \
        hyper_params, len(metric_logger.metrics["episodic_returns"][experiment_name][0])

