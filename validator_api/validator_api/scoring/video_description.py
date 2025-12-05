from validator_api.validator_api.database import get_db_context
from validator_api.validator_api.database.models.focus_video_record import (
    FocusVideoRecord,
)
from validator_api.validator_api.scoring.scoring_service import DetailedVideoDescription
from sqlalchemy import select
from validator_api.validator_api.scoring import focus_scoring_prompts
from validator_api.validator_api.scoring.scoring_service import (
    _make_gemini_request_with_retries,
)


async def get_task_overview(video_id: str) -> str:
    async with get_db_context() as db:
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id, FocusVideoRecord.deleted_at.is_(None)
        )
        result = await db.execute(query)
        video_record = result.scalar_one_or_none()

        if video_record is None:
            raise ValueError(f"Video not found: {video_id}")

        focusing_task = video_record.video_details.get("focusing_task", "")
        focusing_description = video_record.video_details.get(
            "focusing_description", ""
        )

    task_overview = (
        f"# Task Title: {focusing_task}\n\n Task Description:\n{focusing_description}"
    )
    return task_overview


async def get_detailed_video_description(
    video_id: str, task_overview: str
) -> DetailedVideoDescription:
    async with get_db_context() as db:
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id, FocusVideoRecord.deleted_at.is_(None)
        )
        result = await db.execute(query)
        video_record = result.scalar_one_or_none()

        if video_record is None:
            raise ValueError(f"Video not found: {video_id}")

        if (
            video_record.video_details
            and "detailed_video_description" in video_record.video_details
        ):
            return DetailedVideoDescription.model_validate(
                video_record.video_details["detailed_video_description"]
            )

    # Get trajectories data if available
    from validator_api.validator_api.scoring.scoring_service import (
        _get_trajectories_from_video_details,
    )

    trajectories_data = await _get_trajectories_from_video_details(video_id)

    print(f"Trajectories data {'found' if trajectories_data else 'not found'} for video_id: {video_id}")
    if trajectories_data:
        print(f"Trajectories contains {len(trajectories_data.get('events', []))} events")

    description = await _make_gemini_request_with_retries(
        system_prompt=focus_scoring_prompts.DETAILED_DESCRIPTION_SYSTEM_PROMPT,
        user_prompt=focus_scoring_prompts.DETAILED_DESCRIPTION_USER_PROMPT.format(
            task_overview=task_overview
        ),
        video_id=video_id,
        OutputClassSchema=DetailedVideoDescription,
        trajectories_data=trajectories_data,
    )

    # Cache the description in database
    async with get_db_context() as db:
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id, FocusVideoRecord.deleted_at.is_(None)
        )
        result = await db.execute(query)
        video_record = result.scalar_one_or_none()

        if video_record:
            video_details = video_record.video_details or {}
            video_details["detailed_video_description"] = description.model_dump()
            video_record.video_details = video_details
            db.add(video_record)
            await db.commit()

    return description
