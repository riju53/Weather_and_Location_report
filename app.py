import streamlit as st
import sqlite3
import requests

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent


st.set_page_config(
    page_title="AI powered City information and Weather update Assistant",
    page_icon="🌍",
    layout="wide"
)



# For learning/local development you can temporarily use:
# HF_TOKEN = "your_huggingface_token"
# WEATHER_API_KEY = "your_weather_api_key"

HF_TOKEN = st.secrets["HF_TOKEN"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
#WEATHER_API_KEY = "b65988ddae944bf993993731262105"



conn = sqlite3.connect(
    "weather_db.sqlite",
    check_same_thread=False
)

saver = SqliteSaver(conn)



llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    temperature=0.7,
    max_new_tokens=2500,
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(
    llm=llm
)


# WEATHER TOOL

@tool
def weather_update(location: str):
    """
    Generate the current weather report for a given location.
    """

    url = "https://api.weatherapi.com/v1/current.json"

    params = {
        "key": WEATHER_API_KEY,
        "q": location
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return {
        "Location": data["location"]["name"],
        "Region": data["location"]["region"],
        "Country": data["location"]["country"],
        "Local Time": data["location"]["localtime"],
        "Temperature": data["current"]["temp_c"],
        "Wind speed": data["current"]["wind_kph"],
        "Condition": data["current"]["condition"]["text"],
        "Humidity": data["current"]["humidity"],
        "Cloud": data["current"]["cloud"],
        "Feels like": data["current"]["feelslike_c"]
    }


# ASTRONOMY TOOL

@tool
def astronomy(city: str):
    """
    Generate astronomy information for a given location.
    """

    url = "https://api.weatherapi.com/v1/astronomy.json"

    params = {
        "key": WEATHER_API_KEY,
        "q": city
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return {
        "sunrise": data["astronomy"]["astro"]["sunrise"],
        "sunset": data["astronomy"]["astro"]["sunset"],
        "moonrise": data["astronomy"]["astro"]["moonrise"],
        "moonset": data["astronomy"]["astro"]["moonset"]
    }


#  DUCKDUCKGO SEARCH

search = DuckDuckGoSearchRun()


@tool
def detailed_city(location: str):
    """
    Search for detailed information about a location,
    including culture, remarkable places, cost of living,
    lifestyle, and nearby tourist destinations.
    """

    query = f"""
    Search for detailed information about {location}.

    Include:
    1. Culture
    2. Remarkable places
    3. Cost of living
    4. Lifestyle
    5. Nearby tourist destinations

    Provide useful information for a tourist.
    """

    response = search.invoke(query)

    return response


#  SYSTEM PROMPT

system_prompt = """
ROLE:
You are an expert travel assistant specializing in weather,
astronomy, and location information.

CONTEXT:
The user is planning to visit a particular city or location.
Your job is to collect current information using the available tools
and prepare a useful travel report.

TASK:
When the user provides a city or location:

1. First use the weather_update tool.
2. Then use the astronomy tool.
3. Finally use the detailed_city tool.

IMPORTANT:
- You MUST call the tools before generating the final answer.
- Use the user's specified location as the input.
- Do not invent weather or astronomy information.
- Combine all tool results into one useful travel report.
- If no location is provided, ask the user for a location.

OUTPUT:
Provide:
1. Short introduction
2. Weather report
3. Astronomy report
4. Location details
5. Travel summary
"""


#  CREATE LANGGRAPH AGENT

agent = create_react_agent(
    model=model,
    tools=[
        weather_update,
        astronomy,
        detailed_city
    ],
    checkpointer=saver,
    prompt=system_prompt
)


# STREAMLIT UI

st.title("🌍 AI powered City information and Weather update Assistant")

st.write(
    "Get weather, astronomy and detailed location information "
    "for your Location."
)


# SIDEBAR   

with st.sidebar:

    st.header("✈️ AI powered City information and Weather update Assistant")

    st.write(
        "Enter a City and let the AI generate "
        "a complete weather and detaild city report."
    )

    st.divider()

    st.info(
        "The assistant uses:\n\n"
        "🌤️ Weather API\n\n"
        "🌙 Astronomy API\n\n"
        "🔎 DuckDuckGo Search\n\n"
        "🤖 Qwen LLM\n\n"
        "🧠 LangChain"
    )


# USER INPUT

location = st.text_input(
    "📍 Enter your destination",
    placeholder="Example: Kolkata"
)


# GENERATE REPORT

if st.button(
    "🚀 Generate Travel Report",
    use_container_width=True
):

    if not location.strip():

        st.warning(
            "⚠️ Please enter a city or location."
        )

    else:

        with st.spinner(
            f"🔍 Generating detailed report for {location}..."
        ):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"I am going to visit "
                                    f"{location}."
                                )
                            }
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": location
                        }
                    }
                )

                final_answer = response["messages"][-1].content

                st.success(
                    f"✅ Travel report generated for {location}"
                )

                st.markdown("---")

                st.markdown(final_answer)

            except Exception as e:

                st.error(
                    "❌ Something went wrong while "
                    "generating the report."
                )

                st.exception(e)
