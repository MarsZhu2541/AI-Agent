"""Tools the chat agent can call."""

import logging
from pathlib import Path
from typing import Annotated, Any, NamedTuple, Optional

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
import os

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
APIHZ_URL = "https://cn.apihz.cn/api/tianqi/tqybip.php?id={id}&key={key}&ip={ip}&day=1&hourtype=1&suntimetype=1"
# api.open-meteo.com 在当前网络会 TLS 握手超时，预报改走可连通的 wttr.in。
FORECAST_URL = "https://wttr.in/{latitude},{longitude}"
_HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_USER_AGENT = "ai-agent-outfit/1.0"
WEATHER_API_ID = os.getenv("WEATHER_API_ID")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


def _text(value: Any) -> str:
    """Return a trimmed string, or an empty string when the value is missing."""
    if value is None:
        return ""
    return str(value).strip()



async def get_position(query: str) -> dict:
    """根据地区名称获取经纬度。例如 query='武汉'"""
    cleaned = query.strip()
    if not cleaned:
        return {}

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.get(
                GEOCODING_URL,
                params={
                    "name": cleaned,
                    "count": 1,
                    "language": "zh",
                    "format": "json",
                },
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.exception("Geocoding failed, query=%s", query)
        return {}

    results = payload.get("results") or []
    if not results:
        return {}

    first = results[0]
    lat = first.get("latitude")
    lon = first.get("longitude")
    return {"latitude": lat, "longitude": lon}


class WeatherInfo(NamedTuple):
    temperature: float
    apparent_temperature: float
    humidity: int
    precipitation: float
    weather_code: int
    wind_speed: float
    time: str
    weather: str = ""
    place: str = ""
    day_high: str = ""
    day_low: str = ""

    def format(self) -> str:
        desc = self.weather or _weather_code_desc(self.weather_code)
        place = f"{self.place}：" if self.place else ""
        text = (
            f"{place}{desc}，气温 {self.temperature}°C（体感 {self.apparent_temperature}°C），"
            f"湿度 {self.humidity}%，风速 {self.wind_speed} km/h"
        )
        if self.day_high or self.day_low:
            text += f"，白天 {self.day_high}°C / 夜间 {self.day_low}°C"
        return text


# wttr.in 使用 WorldWeatherOnline 天气代码，不是 Open-Meteo 的 WMO 代码。
_WEATHER_CODE_MAP = {
    113: "晴", 116: "局部多云", 119: "多云", 122: "阴",
    143: "薄雾", 176: "局部有雨", 179: "局部有雪", 182: "局部雨夹雪",
    185: "局部冻毛毛雨", 200: "局部雷暴", 227: "吹雪", 230: "暴风雪",
    248: "雾", 260: "冻雾",
    263: "局部小毛毛雨", 266: "小毛毛雨", 281: "冻毛毛雨", 284: "大冻毛毛雨",
    293: "局部小雨", 296: "小雨", 299: "间歇中雨", 302: "中雨",
    305: "间歇大雨", 308: "大雨", 311: "小冻雨", 314: "中到大冻雨",
    317: "小雨夹雪", 320: "中到大雨夹雪",
    323: "局部小雪", 326: "小雪", 329: "局部中雪", 332: "中雪",
    335: "局部大雪", 338: "大雪", 350: "冰粒",
    353: "小阵雨", 356: "中到大阵雨", 359: "暴雨",
    362: "小阵雨夹雪", 365: "中到大阵雨夹雪",
    368: "小阵雪", 371: "中到大阵雪",
    374: "小阵冰雹", 377: "中到大阵冰雹",
    386: "局部雷阵雨", 389: "雷阵雨", 392: "局部雷阵雪", 395: "雷阵雪",
}


def _weather_code_desc(code: int) -> str:
    return _WEATHER_CODE_MAP.get(code, f"未知天气({code})")


async def _search_weather(lat, lon) -> Optional[WeatherInfo]:
    """根据经纬度查询 wttr.in 实时天气。失败返回 None。"""
    try:
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(
                FORECAST_URL.format(latitude=lat, longitude=lon),
                params={"format": "j1"},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.exception(
            "Weather fetch failed, lat=%s lon=%s", lat, lon
        )
        return None

    conditions = payload.get("current_condition") or []
    if not conditions:
        return None
    current = conditions[0]

    return WeatherInfo(
        temperature=float(current["temp_C"]),
        apparent_temperature=float(current["FeelsLikeC"]),
        humidity=int(current["humidity"]),
        precipitation=float(current["precipMM"]),
        weather_code=int(current["weatherCode"]),
        wind_speed=float(current["windspeedKmph"]),
        time=str(current.get("observation_time") or ""),
    )


def _weather_from_ip_payload(payload: dict) -> Optional[WeatherInfo]:
    """把 IP 天气接口的返回转成统一结构。字段和 wttr.in 不一样。"""
    if payload.get("code") != 200:
        logger.error(
            "Weather by IP rejected, code=%s msg=%s",
            payload.get("code"),
            payload.get("msg"),
        )
        return None
    nowinfo = payload.get("nowinfo")
    if not isinstance(nowinfo, dict):
        return None
    # 接口风速是 m/s，统一文案用 km/h。
    wind_kmh = round(float(nowinfo["windSpeed"]) * 3.6, 1)
    return WeatherInfo(
        temperature=float(nowinfo["temperature"]),
        apparent_temperature=float(nowinfo.get("feelst") or nowinfo["temperature"]),
        humidity=int(nowinfo["humidity"]),
        precipitation=float(nowinfo.get("precipitation") or 0),
        weather_code=0,
        wind_speed=wind_kmh,
        time=str(nowinfo.get("uptime") or ""),
        weather=_text(payload.get("weather1")),
        place=_text(payload.get("name") or payload.get("shi")),
        day_high=_text(payload.get("wd1")),
        day_low=_text(payload.get("wd2")),
    )


async def _search_weather_by_ip(ip) -> Optional[WeatherInfo]:
    """根据 IP 查询实时天气。失败返回 None。"""
    try:
        async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(
                APIHZ_URL.format(id=WEATHER_API_ID, key=WEATHER_API_KEY, ip=ip),
            )
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.exception("Weather fetch failed, ip=%s", ip)
        return None

    try:
        return _weather_from_ip_payload(payload)
    except (KeyError, TypeError, ValueError):
        logger.exception("Weather payload unreadable, ip=%s", ip)
        return None


@tool(
    description=(
        "查询指定地区的实时天气。"
        "用户问天气，或需要按天气给穿搭建议时调用。"
        "query 填地区名，例如「武汉」。"
    ),
)
async def get_weather(query: Annotated[str, "用户地区 例如 武汉"]):
    """查询指定地区的实时天气。用户问天气或需要穿搭建议时调用。"""
    logger.info("enter get_weather")
    cleaned = _text(query)
    if not cleaned:
        return "搜索关键词为空"
    position = await get_position(cleaned)
    if not position or position.get("latitude") is None:
        return f"未找到地区：{cleaned}"

    weather = await _search_weather(position["latitude"], position["longitude"])
    if weather is None:
        return f"未能获取 {cleaned} 的天气"
    return weather.format()


@tool(
    description=(
            "通过ip地址查询天气。"
            "用户消息包括ip地址时，需要按天气给穿搭建议时必须调用。"
            "query 填ip地址，例如「10.162.31.1」。"
    ),
)
async def get_weather_by_ip(query: Annotated[str, "用户地区 例如 221.232.108.214"]):
    """查询指定地区的实时天气。用户问天气或需要穿搭建议时调用。"""
    logger.info("enter get_weather by ip")
    cleaned = _text(query)
    if not cleaned:
        return "搜索关键词为空"

    weather = await _search_weather_by_ip(cleaned)
    if weather is None:
        return f"未能获取 {cleaned} 的天气"
    return weather.format()

tools = [get_weather, get_weather_by_ip]