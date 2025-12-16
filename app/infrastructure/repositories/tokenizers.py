import string
import jieba
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from app.infrastructure.repositories.tokenizer_interface import TokenizerInterface

logger = logging.getLogger(__name__)

curr_dir = Path(__file__).resolve().parent
jieba.set_dictionary(str(curr_dir.joinpath('tokenizer_data','dict.txt.big'))) 

class Preprocessor:
    def __init__(self):
        chinese_punctuation = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。《》「」『』【】—…·"
        self.punctuations = set(string.punctuation) | set(chinese_punctuation)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')) | self._load_custom_stopwords()

    def _load_custom_stopwords(self) -> set[str]:
        curr_dir = Path(__file__).resolve().parent
        stopword_path = curr_dir.joinpath('tokenizer_data','stopwords.txt')
        try:
            with open(stopword_path, 'r') as f:
                custom_stopwords = set(line.strip() for line in f if line.strip())
            return custom_stopwords
        except FileNotFoundError:
            logger.warning(f"Custom stopwords file not found at {stopword_path}. Proceeding without custom stopwords.")
            return set()
        
    def remove_punctuation(self, text:str) -> str:
        map_table = str.maketrans('', '', ''.join(self.punctuations))
        return str.translate(text, map_table)

    def remove_stopwords(self, words:list[str]) -> list[str]:
        return [word for word in words if word not in self.stop_words]
    
    def stem_words(self, words:list[str]) -> list[str]:
        return [self.stemmer.stem(w) for w in words]
    

# -- (Chinese) Jieba Tokenizer
class JiebaTokenizer(TokenizerInterface):
    """
    Ref: 
    https://github.com/fxsjy/jieba?tab=readme-ov-file#%E5%85%B6%E4%BB%96%E8%AF%8D%E5%85%B8
    1. https://tsroten.github.io/zhon/api.html#zhon.hanzi.punctuation
    2. https://github.com/fengdu78/machine_learning_beginner/blob/master/deep-learning-with-tensorflow-keras-pytorch/deep-learning-with-keras-notebooks-master/8.1-jieba-word-tokenizer.ipynb
    """
    def __init__(self, preprocessor:Preprocessor = Preprocessor()):
        self.preprocessor = preprocessor

    def tokenize(self, text:str) -> list[str]:
        text = self.preprocessor.remove_punctuation(text)
        # tokens = [token for token in jieba.cut_for_search(text) if token.strip()] 
        tokens = [token for token in jieba.cut(text, cut_all=False) if token.strip()] 
        tokens = [token for token in tokens if token not in self.preprocessor.punctuations]
        tokens = [token for token in tokens if token not in self.preprocessor.stop_words]
        tokens = self.preprocessor.stem_words(tokens)
        return tokens

# -- (English) Word Tokenizer
class WordTokenizer(TokenizerInterface):
    def __init__(self, preprocessor:Preprocessor = Preprocessor()):
        self.preprocessor = preprocessor

    def tokenize(self, text:str) -> list[str]:
        text = self.preprocessor.remove_punctuation(text.lower())
        words = text.split()
        words = self.preprocessor.remove_stopwords(words)
        words = self.preprocessor.stem_words(words)
        return words

  


if __name__ == "__main__":
    text_en = "I received my Ph.D. in Astrophysics from National Taiwan University (NTU) in 2018. Prior to that, I completed a Master’s degree in Astrophysics and undergraduate degree in Physics, both from NTU. Research is fun! Near the end of my Ph.D. studies, I started to explore the fields of machine learning and data science."
    text = " MLOps 相關的管線工具，因此模型部署有時候也只是簡單的點選操作。久聞包含模型推論在內的許多網路服務底下都是以 Kubernetes（K8s）的方式來管理，但過去僅在專案中間接遇到過，本著多了解一層的精神，決定透過實作補充一些相關知識。"
    
    print("---- English word split ----")
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(text_en)
    print(tokens)
    
    print("---- jieba ----")
    tokenizer = JiebaTokenizer()
    tokens = tokenizer.tokenize(text)
    print(tokens)
