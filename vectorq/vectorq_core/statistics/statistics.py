class Statistics:

    def __init__(self):
        self.correct_hits: int = 0
        self.incorrect_hits: int = 0
        self.vector_db_size: int = 0
        self.num_of_request: int = 0
        # TODO
        
    def get_accuracy(self) -> float:
        # TODO
        return None
    
    def update_accuracy(self, is_correct: bool) -> None:
        # TODO
        pass
    
    def get_statistics(self) -> str:
        # TODO
        return ""
    
    def update_statistics(self) -> None:
        # TODO
        pass
