
class Simulator():
 
    def __init__(
        self,
        environment_generator: 'base.EnvironmentGenerator',
        agent: 'agents.Agent',
        view_model: 'view_models.ViewModel',
        num_scenarios: int,
        num_steps: int):
        
        self.environment_generator = environment_generator
        self.agent = agent
        self.view_model = view_model
        self.num_scenarios = num_scenarios
        self.num_steps = num_steps

    def run(self):
        
        # Loop over number of scenarios
        for scenario in range(self.num_scenarios):
            environment = self.environment_generator.new_environment()

            for step in range(self.num_steps):
                action = self.agent.choose_action(environment.state)
                environment.update(action)
                self.agent.learn(environment.state, environment.reward)
                self.view_model.update(
                    scenario_count=scenario,
                    step_count=step,
                    environment=environment,
                    agent=self.agent)
