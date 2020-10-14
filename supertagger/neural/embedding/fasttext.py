import fasttext  # type: ignore

from supertagger.neural.embedding.pretrained import PreTrained


class FastText(PreTrained):
    """Module for fastText (binary) word embedding."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            file_path: str,
            dropout: float = 0,
    ):
        super(FastText, self).__init__(
            self.load_model(file_path),
            dropout=dropout,
        )

    #
    #
    #  -------- load_model -----------
    #
    def load_model(self, file_path):
        return fasttext.load_model(file_path)

    #
    #
    #  -------- embedding_dim -----------
    #
    def embedding_dim(self) -> int:
        return self.model.get_dimension()

    #
    #
    #  -------- embedding_num -----------
    #
    def embedding_num(self) -> int:
        return len(self.model.get_words())
