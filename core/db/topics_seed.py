"""Global curriculum — 5 levels × 5 topics. No userId. Idempotent by slug."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LEVELS = ("A1", "A2", "B1", "B2", "C1")

_CRITERIA = {"minimumGoals": 4, "minimumConversationSeconds": 300}

_FOCUS = {
    "A1": (
        "Beginner A1: basic introductions, simple daily activities, "
        "basic vocabulary, simple present, short answers."
    ),
    "A2": (
        "Elementary A2: longer conversations, past and future, "
        "daily-life situations, basic opinions, more descriptive answers."
    ),
    "B1": (
        "Intermediate B1: detailed conversations, opinions and explanations, "
        "experiences, storytelling, workplace/social communication."
    ),
    "B2": (
        "Upper Intermediate B2: complex discussions, arguments and reasoning, "
        "abstract topics, professional communication, natural flow."
    ),
    "C1": (
        "Advanced C1: fluent discussion, nuanced opinions, complex reasoning, "
        "professional/academic communication, advanced vocabulary."
    ),
}


def _g(key: str, description: str) -> dict[str, str]:
    return {"key": key, "description": description}


def _topic(
    *,
    level: str,
    order: int,
    slug: str,
    title: str,
    description: str,
    goals: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "title": title,
        "slug": slug,
        "description": f"{description} {_FOCUS[level]}",
        "level": level,
        "order": order,
        "goals": goals,
        "completionCriteria": dict(_CRITERIA),
        "isActive": True,
    }


TOPICS: list[dict[str, Any]] = [
    # --- A1 ---
    _topic(
        level="A1",
        order=1,
        slug="a1-introduction",
        title="Introduction",
        description="Help the learner introduce themselves in simple English.",
        goals=[
            _g("name", "User can introduce their name."),
            _g("location", "User can talk about where they live."),
            _g("education_or_work", "User can briefly explain what they study or do."),
            _g("hobbies", "User can talk about basic hobbies."),
            _g("future_goal", "User can talk about a simple future goal."),
        ],
    ),
    _topic(
        level="A1",
        order=2,
        slug="a1-daily-routine",
        title="Daily Routine",
        description="Talk about a normal day using simple present tense.",
        goals=[
            _g("wake_up", "User can say when they wake up."),
            _g("morning", "User can describe a simple morning activity."),
            _g("work_or_study_day", "User can say what they do during the day."),
            _g("evening", "User can describe an evening activity."),
            _g("sleep", "User can say when they go to sleep."),
        ],
    ),
    _topic(
        level="A1",
        order=3,
        slug="a1-family-friends",
        title="Family & Friends",
        description="Talk about people close to the learner in short sentences.",
        goals=[
            _g("family_members", "User can name people in their family."),
            _g("who_they_live_with", "User can say who they live with."),
            _g("a_friend", "User can mention one friend."),
            _g("time_together", "User can say what they do with family or friends."),
            _g("important_person", "User can say who is important to them."),
        ],
    ),
    _topic(
        level="A1",
        order=4,
        slug="a1-hobbies-interests",
        title="Hobbies & Interests",
        description="Share simple free-time activities and likes.",
        goals=[
            _g("hobby", "User can name a hobby they like."),
            _g("when", "User can say when they do it."),
            _g("why_like", "User can give a simple reason they like it."),
            _g("with_whom", "User can say if they do it alone or with others."),
            _g("want_to_try", "User can mention something they want to try."),
        ],
    ),
    _topic(
        level="A1",
        order=5,
        slug="a1-food-drinks",
        title="Food & Drinks",
        description="Talk about meals, likes, and simple food habits.",
        goals=[
            _g("favorite_food", "User can name food they like."),
            _g("meals", "User can talk about breakfast, lunch, or dinner."),
            _g("drinks", "User can mention drinks they like."),
            _g("home_or_out", "User can say if they cook or eat out."),
            _g("dislike", "User can name food they do not like."),
        ],
    ),
    # --- A2 ---
    _topic(
        level="A2",
        order=1,
        slug="a2-neighborhood",
        title="My Neighborhood",
        description="Describe the local area and daily places around them.",
        goals=[
            _g("area", "User can describe their neighborhood in a few sentences."),
            _g("nearby_places", "User can mention shops, parks, or transport nearby."),
            _g("like_or_not", "User can give a simple opinion about the area."),
            _g("people", "User can say what people around them are like."),
            _g("change", "User can mention something they want to change there."),
        ],
    ),
    _topic(
        level="A2",
        order=2,
        slug="a2-shopping-money",
        title="Shopping & Money",
        description="Talk about buying things, prices, and everyday money habits.",
        goals=[
            _g("where_shop", "User can say where they usually shop."),
            _g("recent_buy", "User can talk about something they bought."),
            _g("price", "User can mention cheap vs expensive in simple terms."),
            _g("online_or_store", "User can say if they prefer online or in-store."),
            _g("saving", "User can talk about saving or spending money simply."),
        ],
    ),
    _topic(
        level="A2",
        order=3,
        slug="a2-travel-experiences",
        title="Travel Experiences",
        description="Share a past trip or a place they visited using past tense.",
        goals=[
            _g("place", "User can name a place they visited."),
            _g("when", "User can say when they went."),
            _g("what_happened", "User can describe one thing that happened."),
            _g("feeling", "User can say how they felt about the trip."),
            _g("next_trip", "User can mention a place they want to visit."),
        ],
    ),
    _topic(
        level="A2",
        order=4,
        slug="a2-health-lifestyle",
        title="Health & Lifestyle",
        description="Talk about health habits, sleep, and feeling well.",
        goals=[
            _g("exercise", "User can say if they exercise and how."),
            _g("sleep", "User can talk about sleep habits."),
            _g("food_habit", "User can mention a healthy or unhealthy habit."),
            _g("stress", "User can say what makes them tired or stressed."),
            _g("want_better", "User can name one lifestyle change they want."),
        ],
    ),
    _topic(
        level="A2",
        order=5,
        slug="a2-plans-future",
        title="Plans & Future",
        description="Talk about near-future plans using going to / will.",
        goals=[
            _g("this_week", "User can talk about a plan for this week."),
            _g("weekend", "User can describe a weekend plan."),
            _g("study_or_work_plan", "User can mention a study or work plan."),
            _g("hope", "User can say something they hope will happen."),
            _g("next_year", "User can share a simple longer-term plan."),
        ],
    ),
    # --- B1 ---
    _topic(
        level="B1",
        order=1,
        slug="b1-work-career",
        title="Work & Career",
        description="Discuss work, studies-to-work, and career direction.",
        goals=[
            _g("role", "User can explain what they do or want to do at work."),
            _g("typical_day", "User can describe a typical work or study day."),
            _g("challenge", "User can talk about a challenge at work or job search."),
            _g("skill", "User can mention a skill they are building."),
            _g("career_aim", "User can explain a career aim in a few sentences."),
        ],
    ),
    _topic(
        level="B1",
        order=2,
        slug="b1-education",
        title="Education",
        description="Talk about learning history, school, and what they learned.",
        goals=[
            _g("background", "User can describe their education background."),
            _g("subject", "User can talk about a subject they liked or found hard."),
            _g("teacher_or_method", "User can mention how they learn best."),
            _g("english_learning", "User can explain why they are learning English."),
            _g("next_step", "User can talk about a next learning step."),
        ],
    ),
    _topic(
        level="B1",
        order=3,
        slug="b1-technology",
        title="Technology",
        description="Discuss everyday tech, phones, and how tech helps or hurts.",
        goals=[
            _g("daily_tech", "User can describe how they use technology daily."),
            _g("helpful", "User can explain one way tech helps them."),
            _g("problem", "User can mention a problem with phones or internet."),
            _g("opinion", "User can give an opinion about social media or AI."),
            _g("future_tech", "User can imagine how tech might change their work."),
        ],
    ),
    _topic(
        level="B1",
        order=4,
        slug="b1-travel-culture",
        title="Travel & Culture",
        description="Compare places, customs, and cultural experiences.",
        goals=[
            _g("culture_home", "User can describe a custom from their culture."),
            _g("other_place", "User can compare home with another city or country."),
            _g("food_culture", "User can talk about food as culture."),
            _g("surprise", "User can share something that surprised them while traveling."),
            _g("respect", "User can discuss how to be respectful in a new place."),
        ],
    ),
    _topic(
        level="B1",
        order=5,
        slug="b1-personal-experiences",
        title="Personal Experiences",
        description="Tell a short personal story with feeling and detail.",
        goals=[
            _g("story", "User can tell a personal story with a beginning and end."),
            _g("why_matters", "User can explain why the experience mattered."),
            _g("lesson", "User can say what they learned from it."),
            _g("feeling", "User can describe feelings in more than one word."),
            _g("similar", "User can connect it to something happening now."),
        ],
    ),
    # --- B2 ---
    _topic(
        level="B2",
        order=1,
        slug="b2-society-social-issues",
        title="Society & Social Issues",
        description="Discuss social issues with reasons and examples.",
        goals=[
            _g("issue", "User can name a social issue they care about."),
            _g("why", "User can explain why it matters with a reason."),
            _g("example", "User can give a real or local example."),
            _g("other_view", "User can mention another point of view."),
            _g("action", "User can suggest a possible action or change."),
        ],
    ),
    _topic(
        level="B2",
        order=2,
        slug="b2-business-workplace",
        title="Business & Workplace",
        description="Talk about teams, meetings, and professional situations.",
        goals=[
            _g("teamwork", "User can describe how they work with others."),
            _g("conflict", "User can talk about a workplace disagreement calmly."),
            _g("communication", "User can explain what good workplace communication looks like."),
            _g("pressure", "User can discuss deadlines or pressure."),
            _g("growth", "User can talk about professional growth."),
        ],
    ),
    _topic(
        level="B2",
        order=3,
        slug="b2-media-communication",
        title="Media & Communication",
        description="Discuss news, media trust, and how people communicate.",
        goals=[
            _g("news_habit", "User can say how they get news or information."),
            _g("trust", "User can discuss trust in media."),
            _g("influence", "User can explain how media influences opinions."),
            _g("personal_style", "User can describe their own communication style."),
            _g("misunderstanding", "User can talk about a misunderstanding and how to fix it."),
        ],
    ),
    _topic(
        level="B2",
        order=4,
        slug="b2-environment",
        title="Environment",
        description="Discuss environment, climate, and personal responsibility.",
        goals=[
            _g("local", "User can describe an environmental issue near them."),
            _g("cause", "User can explain a cause in simple reasoned English."),
            _g("habit", "User can mention a personal habit that helps or harms."),
            _g("policy", "User can give an opinion on a rule or policy."),
            _g("tradeoff", "User can discuss a tradeoff (jobs vs environment, cost vs change)."),
        ],
    ),
    _topic(
        level="B2",
        order=5,
        slug="b2-problem-solving",
        title="Problem Solving & Decision Making",
        description="Walk through a decision: options, reasons, outcome.",
        goals=[
            _g("problem", "User can describe a real problem they faced."),
            _g("options", "User can list more than one option."),
            _g("reason", "User can explain why they chose one option."),
            _g("result", "User can say what happened after the decision."),
            _g("better", "User can reflect on what they would do differently."),
        ],
    ),
    # --- C1 ---
    _topic(
        level="C1",
        order=1,
        slug="c1-global-issues",
        title="Global Issues",
        description="Discuss global issues with nuance, not slogans.",
        goals=[
            _g("issue", "User can frame a global issue clearly."),
            _g("stakeholders", "User can mention different groups affected."),
            _g("complexity", "User can show the issue is not one-sided."),
            _g("evidence", "User can support a view with an example or fact-like detail."),
            _g("implication", "User can discuss a longer-term implication."),
        ],
    ),
    _topic(
        level="C1",
        order=2,
        slug="c1-leadership-management",
        title="Leadership & Management",
        description="Talk about leading people, trust, and difficult calls.",
        goals=[
            _g("style", "User can describe a leadership style they respect."),
            _g("example", "User can share a leadership moment (theirs or observed)."),
            _g("feedback", "User can discuss giving or receiving hard feedback."),
            _g("trust", "User can explain how trust is built on a team."),
            _g("failure", "User can talk about a failure without collapsing into cliché."),
        ],
    ),
    _topic(
        level="C1",
        order=3,
        slug="c1-philosophy-ideas",
        title="Philosophy & Ideas",
        description="Explore abstract ideas with examples from real life.",
        goals=[
            _g("idea", "User can introduce an abstract idea in plain English."),
            _g("example", "User can ground it in a personal or social example."),
            _g("counter", "User can consider a counter-argument."),
            _g("uncertainty", "User can admit what they are not sure about."),
            _g("value", "User can connect the idea to a value they hold."),
        ],
    ),
    _topic(
        level="C1",
        order=4,
        slug="c1-advanced-professional",
        title="Advanced Professional Communication",
        description="Practice precise, diplomatic professional English.",
        goals=[
            _g("audience", "User can adapt a message for a professional audience."),
            _g("diplomacy", "User can disagree politely and clearly."),
            _g("summary", "User can summarize a complex situation briefly."),
            _g("ask", "User can make a precise request or proposal."),
            _g("follow_up", "User can describe a professional follow-up."),
        ],
    ),
    _topic(
        level="C1",
        order=5,
        slug="c1-debate-critical-thinking",
        title="Debate & Critical Thinking",
        description="Argue a position, test it, and revise if needed.",
        goals=[
            _g("claim", "User can state a clear claim."),
            _g("support", "User can support it with reasoning."),
            _g("weakness", "User can name a weakness in their own view."),
            _g("rebuttal", "User can respond to a challenge."),
            _g("revise", "User can revise or qualify the claim after discussion."),
        ],
    ),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
