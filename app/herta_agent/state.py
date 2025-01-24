from typing import Any, TypedDict
import operator
from typing_extensions import Annotated

PaperMetadata = dict[str, Any]

class InputState(TypedDict, total=False):
    target_title: str

class OverallState(InputState, total=False):
    target_paper: PaperMetadata
    citations: list[tuple[PaperMetadata, list[str]]]
    sentiments: Annotated[list[tuple[str, PaperMetadata]], operator.add]
    report: str

class EachCitationState(TypedDict, total=False):
    target_paper: PaperMetadata
    paper: PaperMetadata
    contexts: list[str]
