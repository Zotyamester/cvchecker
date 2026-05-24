from pydantic import BaseModel, Field
import pytest
from google.genai import types
from main import FRONTIER_MODEL, gemini, check_cv
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("AzharAli05/Resume-Screening-Dataset", cache_dir=".hf_cache/")
df = ds["train"].to_pandas()


class GradingSummary(BaseModel):
    accept: bool = Field(
        description="True if and only if the result of the test is deemed successful."
    )
    reason: str = Field(
        description="Short description of why the given reason did not match (semantically) the expected reason."
    )


@pytest.mark.parametrize(
    "cv,job_posting,expected_role,expected_decision,expected_reason",
    df[["Resume", "Job_Description", "Role", "Decision", "Reason_for_decision"]]
    .sample(n=1)
    .itertuples(index=False, name=None),
)
def test_cv_check(
    cv: str,
    job_posting: str,
    expected_role: str,
    expected_decision: str,
    expected_reason: str,
):
    result = check_cv(cv, job_posting)

    # Basic sanitization
    decision = "select" if result.decision else "reject"
    assert (
        decision == expected_decision
    ), f"wrong decision: expected {expected_decision}, got: {decision}"

    # LLM-as-a-Judge
    response = gemini.models.generate_content(
        model=FRONTIER_MODEL,
        contents=f"""
            You are an impartial grading AI. Your job is to evaluate if the system under test produced the correct results, and generate a grading summary accordingly.
            Be lenient. Accept any reason provided it has a slight semantic similarity to the expected one.
            That is,
                1) the deduced role matches (semantically) the expected role.
                2) the given reason matches (semantically) the expected reason.

            Deduced role: {result.role}
            Expected role: {expected_role}

            Given reason: {result.suitability}
            Expected reason: {expected_reason}
        """,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_schema=GradingSummary,
        ),
    )

    grading_summary: GradingSummary = response.parsed  # type: ignore
    if grading_summary is None:
        raise Exception("Failed to parse the grading summary")
    assert grading_summary.accept, f"wrong reason: {grading_summary.reason}"
