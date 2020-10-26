import random 

class RandomSearch():
    def __init__(self, full_trainer, options):
        self.trainer = full_trainer
        self.options = options
        self.results = list()
    
    def choose(self):
        choice = dict()
        for k,v in self.options.items():
            choice[k] = random.choice(v)
        return choice
    
    def fit(self, count = None):
        i = 0
        while (True):
            i += 1
            if i > count:
                break
            try:
                choice = self.choose()
                result = self.trainer.train(**choice)
                self.results.append(result)
            except RuntimeError as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                else:
                    print("####")
                    print(f"Failed: {choice}")
                    print("###")
    
        
        