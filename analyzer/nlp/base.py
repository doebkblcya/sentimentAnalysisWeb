from abc import ABC, abstractmethod

class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> dict:
        """
        分析文本情感
        :param text: 输入文本
        :return: {
            'sentiment': int,  # 1: 正面, 0: 中性, -1: 负面
            'positive_score': float,  # 正面情感分数
            'negative_score': float   # 负面情感分数
        }
        """
        pass