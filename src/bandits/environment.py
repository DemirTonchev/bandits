import numpy as np
from bandits.base import base_rng, check_random_state, random_argmax
from bandits.arms import Arm

class SimpleEnvironment:
    """Simple environment for N-armed bandit. """

    def __init__(self, arms: list[Arm], seed=None):
        """Create the environment. 
        Params:
            arms - list: list of arms objects
        """
        self.arms = arms
        self.rng = check_random_state(seed)
        self.k_arms = len(self.arms)

        self.means = np.asarray([arm.mean for arm in arms])
        self.optimal_arm = np.argmax(self.means)
        self.optimal_mean = self.means[self.optimal_arm]

        self._observation = 0

    def get_stochastic_reward(self, arm_index: int):
        """returns a stochastic reward after pulling(acting) arm = arm_index. This reward depends on the distsribtion of the arm.
        """
        self.reward = self.arms[arm_index].pull()
        return self.reward
    
    @property
    def observation(self):
        return self._observation
    
    @observation.setter
    def observation(self, new_value):
        self._observation = new_value

    # def get_optimal_mean(self):
    #     """Returns the value of the optimal(real) mean
    #     """
    #     return self.optimal_mean

    # def get_optimal_arm(self):
    #     """Returns the index of the optimal mean
    #     """
    #     return self.optimal_arm

    # def get_all_means(self):
    #     return self.means
    
    #gymnasium api
    def step(self, arm_index: int):
        """making stepts to comply with gymnasium api. 
        In bandit problems, an agent selects an action, it receives a reward, the action does not affect the next observation.
        The environment has no dynamics, so the reward is only influenced by the current action. The environment does not evolve along 
        the time dimension, and there is no sequential decision making as in RL, no delayed rewards. 
        Thus the episode terminates afte one step. So Terminated is True after 1 step.
        """
        observation = 0
        info = {}
        terminated = True
        truncated = False
        reward = self.get_stochastic_reward(arm_index)
        return observation, reward, terminated, truncated, info
    
    #gymnasium api
    def reset(self):
        self.info = {'reset': True}
        return 0, self.info

    def __str__(self):
        return f'{type(self).__name__} with arms: {self.arms}'

class ContextualEnvironment:

    def __init__(self, arms, add_bias = True, seed=None, context_generator=None):
        
        self.rng = check_random_state(seed)
        
        self.arms = arms
        self.k_arms = len(arms)
        self.dimension = arms[0].dimension
        self.thetas = [arm.theta for arm in arms]
        self.add_bias = add_bias
        self.context_generator = context_generator

        self.current_optimal_mean = None
        self.current_optimal_arm = None
        self.current_means = None
        self.current_rewards = None
        self.current_context = None
    

    def generate_context(self):
        """Generates context vector and computes current
        real reward probability
        """
        if callable(self.context_generator):
            context = self.context_generator()
        else:
            context = []
            context_vector = self.rng.uniform(0, 1, size=self.dimension - (1 if self.add_bias else 0) ).round(2)
            if self.add_bias:
                context_vector = np.append([1], context_vector)
            # v momenta context e ednakyv, no moje i trqbwa da se dobavi informaciqta ot action a_i, w drug
            # obekt koito ima generate_context(self, arms_context) i da se concat kym tozi 
            context = [context_vector for _ in range(self.k_arms)]
        self.current_context = context
        return context

    def get_stochastic_reward(self, arm_index):
        # pull all arms to generate current mean, rewards and get the optimal one
        # agent/policy/algorithm knows only the context but true means are not revealed
        self.current_rewards = [arm.pull(context_vector) for arm, context_vector in zip(self.arms, self.current_context)]
        self.current_means = [arm.get_current_mean() for arm in self.arms]
        self.current_optimal_arm = random_argmax(self.current_means)
        self.current_optimal_mean = self.current_means[self.current_optimal_arm]
        return self.current_rewards[arm_index]

class SimpleContextualEnvironment(ContextualEnvironment):
    """Simple env 2 states (context) and n arms. 
    Possible observations are: 
    [1.0, 1.0] and [1.0, -1.0] if there is bias term otherwise [1.0] and [-1.0], this is the "current context".
    Rewards depend on the parameters of the arms(given the context)
    """

    def __init__(self, arms, add_bias=True, seed=None, context_generator=None):
        super().__init__(arms, add_bias, seed, context_generator)
        assert self.dimension <= 2, "This environmet is for testing purposes only and allows arms with max dim = 2!"

    def generate_context(self):
        """Generate very simple context
        """
        context_vector = np.array([self.rng.choice([-1.0, 1.0])])
        if self.add_bias:
                context_vector = np.append([1], context_vector)
        self.current_context = [context_vector for _ in range(self.k_arms)]
        return self.current_context
    

    def ucb_max_regret(T, environment):
        deltas = environment.optimal_mean - environment.means
        deltas = deltas[np.nonzero(deltas)]
        return 8*np.log(T)*np.sum(1/deltas) + (1+(np.pi**2)/3)*np.sum(deltas)
