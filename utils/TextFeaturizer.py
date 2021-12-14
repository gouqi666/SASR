from config import lm_token_path


class TextFeaturizer:
    def __init__(self, token_path):
        # token格式：每行一个单词，最开始两个是开始和结束符号，最后一行为#，读取到此停止
        self.word2token = {}
        self.token2word = {}
        self.init_dict(token_path)

    def init_dict(self, token_path):
        with open(token_path, 'r', encoding='utf8') as fp:
            lines = fp.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if line != "#":
                    self.word2token[line] = idx
                    self.token2word[idx] = line
                else:
                    self.word2token[""] = idx
                    self.token2word[idx] = ""
                    break

    @property
    def vocabulary(self):
        return list(self.word2token.keys())

    @property
    def vocab_size(self):
        return len(self.token2word)

    def encode(self, words):
        return [self.word2token[word] for word in words]

    def decode(self, tokens):
        return [self.token2word[token] for token in tokens]


if __name__ == '__main__':
    print(TextFeaturizer(lm_token_path).encode("我是大帅哥"))

