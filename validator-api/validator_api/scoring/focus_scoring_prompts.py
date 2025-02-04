# TASK_VALUATION_CRITERIA = """The kind of tasks that we want to see:
# - Tasks that contribute to scientific discovery or AI advancement
# - Creative acts that result in the creation of something new
# - Tasks that demonstrate Chain of Thought (CoT) and are useful for training AI
# - High novelty in approach or outcome
# - Tasks that current AI systems struggle with
# - Videos of coding, AI research, or solving AI engineering problems
# - Application of the scientific process, including designing and implementing experiments
# - Tasks that seek more knowledge or demonstrate critical thinking and creation
# - Students learning challenging new material
# - Office workers efficiently completing assigned work

# The kind of tasks that we don't want to see:
# - Extremely mundane, boring, or repetitive tasks
# - Tasks that can be easily completed by existing AI systems (e.g., basic copywriting)
# - Tasks already present in existing datasets"""

# TASK_SCORE_SYSTEM_PROMPT = f"""
# You are an AI tasked with evaluating proposed tasks for a cryptocurrency reward system. The goal is to encourage tasks that contribute to scientific discovery, AI advancement, creativity, education, and productivity while avoiding repetitive, busywork, or unproductive tasks.

# Here are the criteria for tasks:

# {TASK_VALUATION_CRITERIA}

# You will evaluate the task based on this rubric:
# - Relevance: How well does the task align with what we want and avoid what we don't want?
# - Impact: How significant is the task's potential contribution to our goals?
# - Feasibility: Is the task realistic, achievable, and well-defined?
# - Efficiency: Does the task make good use of resources in pursuing our objectives?
# """

# TASK_SCORE_USER_PROMPT = """
# Here is the task to evaluate:

# <task_description>
# {task_overview}
# </task_description>

# Analyze this task based on the provided criteria and rubric. Consider both positive and negative aspects, and explain your thought process thoroughly.

# Provide your reasoning for why the task is or is not a good fit for the goal. Discuss how it aligns with or deviates from the criteria for what we want and don't want. Evaluate its potential impact, feasibility, and efficiency.

# After providing your reasoning, assign a score between 0.0 and 1.0 to indicate how well the task fits our goals. Use this scale:
# - 0.0-0.2: Poor fit, largely irrelevant or counterproductive
# - 0.2-0.4: Weak fit, minimal contribution to the goal
# - 0.4-0.6: Moderate fit, somewhat helpful but not ideal
# - 0.6-0.8: Good fit, clearly contributes to the goal
# - 0.8-1.0: Excellent fit, highly effective in achieving the goal

# Remember to adhere to the JSON schema provided.
# """

DETAILED_DESCRIPTION_SYSTEM_PROMPT = """
You are tasked with watching a screen recording of a human performing a task and creating a detailed annotation of the process. Your goal is to produce a description so thorough and precise that another human or AI could replicate the user's step-by-step sequence without ever seeing the video.

After watching the video, you will create an annotation following the DetailedVideoDescription schema. This schema includes four main components: applications_used, completion_sequence_steps, user_feedback, and description.

For each component of the schema, follow these guidelines:

1. applications_used: List all software applications, websites, or tools used in the video.

2. completion_sequence_steps: Provide a highly detailed, step-by-step breakdown of the entire process. Each step should be clear, concise, and actionable. Include any relevant details that can be gleaned from the screen recording. Number each step for clarity.

3. user_feedback: Offer constructive feedback to the user on their performance. Highlight areas where they excelled and suggest potential improvements or more efficient methods.

4. description: Write a high-level summary of the video content, capturing the essence of the task and its execution in a few sentences.

When writing your annotation, be as precise and detailed as possible. Imagine that someone reading your description should be able to replicate the exact actions without ever seeing the original video. Pay special attention to any novel or highly interesting aspects of the video. Detail such aspects more thoroughly.
"""

DETAILED_DESCRIPTION_USER_PROMPT = """
Watch the provided video carefully, paying close attention to every action taken by the user. Take note of the applications used, the sequence of steps performed, and any notable techniques employed.

Note that the user is completing a task that is described as follows:

<task_description>
{task_overview}
</task_description>

Then, write a detailed description based on the criteria outlined. Remember to focus especially on the task completion sequence and any novel or highly interesting aspects of the video.

Remember to be thorough, clear, and precise in your annotation. Your goal is to create a description that allows for perfect replication of the task.

Remember to adhere to the JSON schema provided.
"""

# VIDEO_SCORING_SYSTEM_PROMPT = f"""
# You are an expert in evaluating task completion based on video recordings.
# Your role is to analyze a screen recording of a user performing a task and provide a detailed breakdown of their performance, focusing on how well they completed the assigned task.

# You will be provided with:
# 1. A task overview describing the assigned task.
# 2. The screen recording video of the user performing the task.
# 3. A detailed description of the video content.

# Your goal is to evaluate the user's performance and provide a completion score following the CompletionScore schema.
# This schema includes a final score and a rationale.

# For each component of the schema, follow these guidelines:

# 1. reasoning_steps: Provide a list of logical steps you took to arrive at your final score. Each step should be prefixed with "Step X: " where X is the step number. Start by first reiterating the task overview and what some steps might look like to complete the task.

# 2. focus_score: Evaluate how focused the user was on completing the task, based on their actions. Score between 0.0 and 1.0.

# 3. educational_score: Assess how clear the user's steps are and how easy it is to follow along. Score between 0.0 and 1.0.

# 4. completion_score: Assess how well the user completed the task, considering their focus, distraction level, and how quickly they completed the task, relative to the task's difficulty. Score between 0.0 and 1.0.

# 5. creativity_score: Assess how creative the user's approach to the task was. Score between 0.0 and 1.0.

# 6. final_score: Calculate an overall completion score based on your evaluation. Score between 0.0 and 1.0.

# 7. rationale: Provide a concise explanation for the given completion score.

# Be thorough and objective in your evaluation, considering all aspects of the user's performance as described in the video description.

# Note that not all tasks are created equal. When evaluating the task completion, keep in mind the following criteria:

# {TASK_VALUATION_CRITERIA}

# Prioritize higher scores for tasks and completions that align with what we want to see, and lower scores for those that align with what we don't want to see.
# """

# VIDEO_SCORING_USER_PROMPT = """
# Based on the task description and video provided, please provide a completion score breakdown. Evaluate how well the user completed the assigned task, considering their focus, the novelty of their approach, and overall effectiveness.

# <task_description>
# {task_overview}
# <task_description>
# {detailed_video_description_string}
# Use the following rubric to assign the focus_score:
# - 0.0-0.2: Poor focus, distractions completely derail the task
# - 0.2-0.4: Weak focus, distractions meaningfully affect the task but are overcome
# - 0.4-0.6: Moderate focus, distractions are a minor inconvenience
# - 0.6-0.8: Good focus, little to no distractions
# - 0.8-1.0: Excellent focus, the user is completely engrossed in the task, in a flow state

# Use the following rubric to assign the educational_score:
# - 0.0-0.2: Poor educational quality, the user's steps are unclear or difficult to follow
# - 0.2-0.4: Weak educational quality, the user's steps can be vageuly followed
# - 0.4-0.6: Moderate educational quality, the user's steps are clear and easy to follow
# - 0.6-0.8: Good educational quality, the user's steps are clear and easy to follow
# - 0.8-1.0: Excellent educational quality, the user's steps are clear and easy to follow

# Use the following rubric to assign the creativity_score:
# - 0.0-0.2: Poor creativity, the user's approach is unoriginal or uninteresting, not even enough to get the job done
# - 0.2-0.4: Weak creativity, the user manages to get the job done but it's not very interesting or creative
# - 0.4-0.6: Moderate creativity, the user's approach is original and creative
# - 0.6-0.8: Good creativity, the user's approach is highly creative and innovative
# - 0.8-1.0: Excellent creativity, the user's approach is groundbreaking and entirely novel

# Use the following rubric to assign the completion_score:
# - 0.0-0.2: Poor task completion, largely irrelevant or counterproductive
# - 0.2-0.4: Weak task completion, minimal contribution to the goal
# - 0.4-0.6: Moderate task completion, somewhat helpful but not ideal
# - 0.6-0.8: Good task completion, the task was diligently completed
# - 0.8-1.0: Excellent task completion, the task was completed with high quality and efficiency

# For the final_score, use your best judgment to assign a score between 0.0 and 1.0 in light of the reasoning_steps, focus_score, educational_score, creativity_score, and completion_score.

# Remember to adhere to the JSON schema provided for the CompletionScore.
# """

TASK_COMPLETION_SYSTEM_PROMPT = """
You are an expert in evaluating task completion based on video recordings.
Your role is to analyze a screen recording of a user performing a task and provide a detailed breakdown of their performance, focusing on how well they completed the assigned task.
Ignore the OMEGA Focus distraction notifications that may appear on the top right of the user's screen.
The content of these notifications should not be factored into your evaluation.

You will be provided with:
1. A task overview describing the assigned task.
2. The screen recording video of the user performing the task.
3. Detailed description of the user's actions in the video.

Your goal is to evaluate the user's performance and provide a completion score following the CompletionScore schema.
This schema includes a final score and a rationale.
In the rationale, try to reference specific guidelines from the task overview/description to justify your score.
"""

TASK_COMPLETION_USER_PROMPT = """
Based on the provided completion sequence steps and video provided, please provide a completion score breakdown.
Evaluate how well the user completed the assigned task, considering their focus and overall effectiveness.
Please use the task description to evaluate the user's performance, which may include specific steps needed to complete the task.
Ignore the OMEGA Focus distraction notifications that may appear on the top right of the user's screen.
EXTREMELY IMPORTANT: Again, the content of these distraction notifications should NOT be factored into your evaluation.

This is the task overview:
<task_overview>
{task_overview}
</task_overview>

This is the detailed description of the user's actions in the video, to aid you in your evaluation:
<completion_sequence_steps>
{completion_sequence_steps}
</completion_sequence_steps>

If the user accomplishes the spirit of the task according to the task title, but does not complete it exactly as described according to the task description, you should still award some score (not 0.0).

Use the following rubric to assign the completion_score:
- 0.0-0.2: Poor task completion, largely irrelevant or counterproductive
- 0.2-0.4: Weak task completion, minimal completion towards the goal
- 0.4-0.6: Moderate task completion, somewhat helpful but not ideal, maybe the user was distracted or did not follow the task description
- 0.6-0.8: Good task completion, the task was diligently completed
- 0.8-1.0: Excellent task completion, the task was completed with high quality and efficiency
"""

# these are for scoring on only the annotated text description, without any video analysis
DESC_ONLY_TASK_COMPLETION_SYSTEM_PROMPT = """You are an expert in analyzing task performance videos, with three distinct phases of analysis.

You will be provided with:
1. Task overview describing the user's assigned task
2. Detailed description and annotated transcript of the user's screen recording of their task completion

Analyze the annotated transcript to provide helpful, actionable feedback.

Provide meaningful feedback including:
- Completion score following CompletionScore schema
- Detailed feedback referencing task guidelines
- Specific strengths and areas for improvement
- Ignore OMEGA Focus notifications in top right

Scoring rubric:
- 0.0-0.2: Poor task completion, largely irrelevant or counterproductive
- 0.2-0.4: Weak task completion, minimal completion towards the goal
- 0.4-0.6: Moderate task completion, somewhat helpful but not ideal, maybe the user was distracted or did not follow the task description
- 0.6-0.8: Good task completion, the task was diligently completed
- 0.8-1.0: Excellent task completion, the task was completed with high quality and efficiency

OUTPUT JSON FORMAT:
{
    "rationale": "Detailed explanation of how well the user completed the task, including specific strengths and areas for improvement",
    "completion_score": float between 0.0 and 1.0
}
"""

DESC_ONLY_TASK_COMPLETION_USER_PROMPT = """Based on the provided annotated transcript, please provide a completion score breakdown.
Evaluate how well the user completed the assigned task, considering their focus and overall effectiveness.
Please use the task description to evaluate the user's performance, which may include specific steps needed to complete the task.

This is the task overview:
<task_overview>
{task_overview}
</task_overview>

This is a list of some of the applications used by the user:
<applications_used>
{applications_used}
</applications_used>

This is the detailed description of the user's actions in the video:
<annotated_transcript>
{completion_sequence_steps}
</annotated_transcript>

If the user accomplishes the spirit of the task according to the task title, but does not complete it exactly as described according to the task description, you should still award some score (not 0.0)."""
