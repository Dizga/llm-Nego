
def train_agent_ppo(self, 
                    agent,
                    ppo_args,
                    folder_path
                    ):

    # Extract training dataset from folder raw data
    queries, responses, scores = self.logger.extract_hf_ppo_dataset(folder_path, p0=True)
    queries_p1, responses_p1, scores_p1 = self.logger.extract_hf_ppo_dataset(folder_path, p0=False)
    queries.append(*queries_p1)
    responses.append(*responses_p1)
    scores.append(*scores_p1)

    # Initiate training 
    agent.init_ppo_trainer(ppo_args)
    agent.