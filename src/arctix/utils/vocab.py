r"""Contain the vocabulary class."""

from __future__ import annotations

__all__ = ["Vocabulary"]

import logging
from collections import Counter
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from coola import objects_are_equal
from coola.equality.comparators import BaseEqualityComparator
from coola.equality.handlers import EqualHandler, SameObjectHandler, SameTypeHandler
from coola.equality.testers import EqualityTester

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Hashable)


class Vocabulary(Generic[T]):
    r"""Implement a vocabulary built from a counter of tokens.

    Args:
        counter: The counter used to generate the vocabulary.
            The order of the items in the counter is used to define
            the index-to-token and token-to-index mappings.

    Example usage:

    ```pycon

    >>> from collections import Counter
    >>> from arctix.utils.vocab import Vocabulary
    >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
    >>> vocab.get_index("a")
    1

    ```
    """

    def __init__(self, counter: Counter) -> None:
        self._counter = counter
        self._index_to_token = tuple(self._counter.keys())
        self._token_to_index = {token: i for i, token in enumerate(self._index_to_token)}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  counter={self.counter},\n"
            f"  index_to_token={self.get_index_to_token()},\n"
            f"  token_to_index={self.get_token_to_index()},\n"
            f")"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(vocab_size={self.get_vocab_size():,})"

    def __len__(self) -> int:
        return len(self._counter)

    @property
    def counter(self) -> Counter:
        r"""The counter of the vocabulary."""
        return self._counter

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two vocabularies are equal or not.

        Args:
            other: The value to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the vocabularies are equal,
                ``False`` otherwise.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab.equal(Vocabulary(Counter({"b": 4, "a": 1, "c": 2})))
        False

        ```
        """
        if not isinstance(other, Vocabulary):
            return False
        return (
            objects_are_equal(self.counter, other.counter, equal_nan=equal_nan)
            and objects_are_equal(self.get_index_to_token(), other.get_index_to_token())
            and objects_are_equal(self.get_token_to_index(), other.get_token_to_index())
        )

    def get_index(self, token: T) -> int:
        r"""Return the index for a given index.

        Args:
            token: The token.

        Returns:
            The token index.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab.get_index("a")
        1
        >>> vocab.get_index("b")
        0
        >>> vocab.get_index("c")
        2

        ```
        """
        return self._token_to_index[token]

    def get_index_to_token(self) -> tuple[T, ...]:
        r"""Return the index to token mapping.

        Returns:
            The index to token mapping.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab.get_index_to_token()
        ('b', 'a', 'c')

        ```
        """
        return self._index_to_token

    def get_token(self, index: int) -> T:
        r"""Return the token for a given index.

        Args:
            index: The index.

        Returns:
            The token associated to the index.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab.get_token(0)
        'b'
        >>> vocab.get_token(1)
        'a'
        >>> vocab.get_token(2)
        'c'

        ```
        """
        return self._index_to_token[index]

    def get_token_to_index(self) -> dict[T, int]:
        r"""Return the token to index mapping.

        Returns:
            The token to index mapping.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab.get_token_to_index()
        {'b': 0, 'a': 1, 'c': 2}

        ```
        """
        return self._token_to_index

    def get_vocab_size(self) -> int:
        r"""Return the vocabulary size.

        Returns:
            The vocabulary size.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab.get_vocab_size()
        3

        ```
        """
        return len(self)

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Load a state dict to the current vocabulary.

        Args:
            state_dict: The state dict to load.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({}))
        >>> vocab.state_dict()
        {'counter': Counter(), 'index_to_token': (), 'token_to_index': {}}
        >>> vocab.load_state_dict(
        ...     {
        ...         "index_to_token": ("b", "a", "c"),
        ...         "token_to_index": {"b": 0, "a": 1, "c": 2},
        ...         "counter": Counter({"b": 3, "a": 1, "c": 2}),
        ...     }
        ... )
        >>> vocab.state_dict()
        {'counter': Counter({'b': 3, 'c': 2, 'a': 1}),
         'index_to_token': ('b', 'a', 'c'),
         'token_to_index': {'b': 0, 'a': 1, 'c': 2}}

        ```
        """
        self._counter = state_dict["counter"]
        self._index_to_token = state_dict["index_to_token"]
        self._token_to_index = state_dict["token_to_index"]

    def state_dict(self) -> dict:
        r"""Return the state dict of the vocabulary.

        Returns:
            The state dict which contains 3 keys: ``"counter"``,
                ``"index_to_token"``, and ``"token_to_index"``.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab.state_dict()
        {'counter': Counter({'b': 3, 'c': 2, 'a': 1}),
         'index_to_token': ('b', 'a', 'c'),
         'token_to_index': {'b': 0, 'a': 1, 'c': 2}}

        ```
        """
        return {
            "counter": self._counter,
            "index_to_token": self._index_to_token,
            "token_to_index": self._token_to_index,
        }

    def add(self, other: Vocabulary) -> Vocabulary:
        r"""Create a new vocabulary where elements from ``other`` are
        added to ``self``.

        Args:
            other: The vocabulary to add.

        Returns:
            The new vocabulary.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab1 = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab2 = Vocabulary(Counter({"b": 3, "d": 7}))
        >>> vocab = vocab1.add(vocab2)
        >>> vocab.counter
        Counter({'d': 7, 'b': 6, 'c': 2, 'a': 1})
        >>> vocab.get_index_to_token()
        ('b', 'a', 'c', 'd')

        ```
        """
        return Vocabulary(self.counter + other.counter)

    def sub(self, other: Vocabulary) -> Vocabulary:
        r"""Create a new vocabulary where elements from ``other`` are
        removed from ``self``.

        Args:
            other: The vocabulary to subtract.

        Returns:
            The new vocabulary.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab1 = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
        >>> vocab2 = Vocabulary(Counter({"b": 3, "d": 7}))
        >>> vocab = vocab1.sub(vocab2)
        >>> vocab.counter
        Counter({'c': 2, 'a': 1})
        >>> vocab.get_index_to_token()
        ('a', 'c')

        ```
        """
        return Vocabulary(self.counter - other.counter)

    def sort_by_count(self, descending: bool = True) -> Vocabulary:
        r"""Create a new vocabulary where the counter is sorted by count.

        If multiple tokens have the same count, they are sorted by
        token values.

        Args:
            descending: If ``True``, the items are sorted in
                descending order by token.

        Returns:
            The new vocabulary where the counter is sorted by count.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_count()
        >>> vocab.counter
        Counter({'b': 3, 'c': 2, 'a': 1})
        >>> vocab.get_index_to_token()
        ('b', 'c', 'a')

        ```
        """
        return Vocabulary(
            Counter(
                dict(
                    sorted(
                        self.counter.items(),
                        key=lambda item: (item[1], item[0]),
                        reverse=descending,
                    )
                )
            )
        )

    def sort_by_token(self, descending: bool = False) -> Vocabulary:
        r"""Create a new vocabulary where the counter is sorted by token.

        Args:
            descending: If ``True``, the items are sorted in
                descending order by token.

        Returns:
            The new vocabulary where the counter is sorted by token.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_token()
        >>> vocab.counter
        Counter({'b': 3, 'c': 2, 'a': 1})
        >>> vocab.get_index_to_token()
        ('a', 'b', 'c')

        ```
        """
        return Vocabulary(Counter(dict(sorted(self.counter.items(), reverse=descending))))

    def most_common(self, max_num_tokens: int) -> Vocabulary:
        r"""Get a new vocabulary with the ``max_num_tokens`` most common
        tokens of the current vocabulary.

        Args:
            max_num_tokens: The maximum number of tokens.

        Returns:
            The new vocabulary with the most common tokens.
                The counter is sorted by decreasing order of count.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).most_common(2)
        >>> vocab.counter
        Counter({'b': 3, 'c': 2})
        >>> vocab.get_index_to_token()
        ('b', 'c')

        ```
        """
        return Vocabulary(Counter(dict(self.counter.most_common(max_num_tokens))))

    @classmethod
    def from_token_to_index(cls, token_to_index: dict[str, int]) -> Vocabulary:
        r"""Instantiate a ``Vocabulary`` from a token to index mapping.

        The counter is initialized to 1 for each token.

        Args:
            token_to_index: The token to index mapping.

        Returns:
            The instantiated ``Vocabulary``.

        Example usage:

        ```pycon

        >>> from collections import Counter
        >>> from arctix.utils.vocab import Vocabulary
        >>> vocab = Vocabulary.from_token_to_index({"grizz": 2, "polar": 0, "bear": 1})
        >>> vocab
        Vocabulary(
          counter=Counter({'polar': 1, 'bear': 1, 'grizz': 1}),
          index_to_token=('polar', 'bear', 'grizz'),
          token_to_index={'polar': 0, 'bear': 1, 'grizz': 2},
        )
        >>> vocab.get_token_to_index()
        {'polar': 0, 'bear': 1, 'grizz': 2}

        ```
        """
        mapping = dict(sorted(token_to_index.items(), key=lambda item: item[1]))
        vocab = Vocabulary(Counter({token: 1 for token in mapping}))
        if not objects_are_equal(vocab.get_token_to_index(), token_to_index):
            msg = (
                "token_to_index and the vocabulary token to index mapping do not match:\n"
                f"{token_to_index}\n{vocab.get_token_to_index()}"
            )
            raise RuntimeError(msg)
        return vocab


class VocabularyEqualityComparator(BaseEqualityComparator[Vocabulary]):
    r"""Implement an equality comparator for ``Vocabulary`` objects."""

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> VocabularyEqualityComparator:
        return self.__class__()

    def equal(self, actual: Vocabulary, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


if not EqualityTester.has_comparator(Vocabulary):  # pragma: no cover
    EqualityTester.add_comparator(Vocabulary, VocabularyEqualityComparator())
