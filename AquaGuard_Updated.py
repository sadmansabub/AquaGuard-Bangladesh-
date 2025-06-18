# AquaGuard Bangladesh - Enhanced Disaster Resilience Water Management Platform
# Complete Python Implementation with Flask, GIS Integration, and Real-time User Reporting

from flask import Flask, render_template, jsonify, request
import json
import random
import time
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import threading
import sqlite3
import os
import sys
import io
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'aquaguard_bangladesh_secret_key_2024'

# Bangladesh-specific coordinates and locations
BANGLADESH_BOUNDS = {
    'north': 26.6382,
    'south': 20.3756,
    'east': 92.6804,
    'west': 88.0075
}

BANGLADESH_DISTRICTS = [
    {'name': 'Dhaka', 'lat': 23.8103, 'lng': 90.4125, 'division': 'Dhaka'},
    {'name': 'Chittagong', 'lat': 22.3569, 'lng': 91.7832, 'division': 'Chittagong'},
    {'name': 'Sylhet', 'lat': 24.8949, 'lng': 91.8687, 'division': 'Sylhet'},
    {'name': 'Rajshahi', 'lat': 24.3745, 'lng': 88.6042, 'division': 'Rajshahi'},
    {'name': 'Khulna', 'lat': 22.8456, 'lng': 89.5403, 'division': 'Khulna'},
    {'name': 'Barisal', 'lat': 22.7010, 'lng': 90.3535, 'division': 'Barisal'},
    {'name': 'Rangpur', 'lat': 25.7439, 'lng': 89.2752, 'division': 'Rangpur'},
    {'name': 'Mymensingh', 'lat': 24.7471, 'lng': 90.4203, 'division': 'Mymensingh'},
    {'name': 'Cox\'s Bazar', 'lat': 21.4272, 'lng': 92.0058, 'division': 'Chittagong'},
    {'name': 'Gazipur', 'lat': 23.9999, 'lng': 90.4203, 'division': 'Dhaka'},
    {'name': 'Cumilla', 'lat': 23.4607, 'lng': 91.1809, 'division': 'Chittagong'},
    {'name': 'Jessore', 'lat': 23.1634, 'lng': 89.2182, 'division': 'Khulna'},
    {'name': 'Bogura', 'lat': 24.8465, 'lng': 89.3775, 'division': 'Rajshahi'},
    {'name': 'Pabna', 'lat': 24.0064, 'lng': 89.2372, 'division': 'Rajshahi'},
    {'name': 'Tangail', 'lat': 24.2513, 'lng': 89.9164, 'division': 'Dhaka'},
]

FLOOD_PRONE_AREAS = [
    {'name': 'Brahmaputra Basin', 'lat': 25.5, 'lng': 89.5, 'risk_level': 'High'},
    {'name': 'Ganges Delta', 'lat': 23.5, 'lng': 90.5, 'risk_level': 'Very High'},
    {'name': 'Meghna River', 'lat': 23.8, 'lng': 90.8, 'risk_level': 'High'},
    {'name': 'Padma River', 'lat': 23.9, 'lng': 89.5, 'risk_level': 'Medium'},
    {'name': 'Jamuna River', 'lat': 24.5, 'lng': 89.8, 'risk_level': 'High'},
]


# Enhanced Data Models
@dataclass
class Alert:
    id: int
    type: str  # 'flood', 'drought', 'cyclone', 'waterlogging'
    severity: str  # 'Low', 'Medium', 'High', 'Critical'
    location: str
    lat: float
    lng: float
    timestamp: str
    description: str = ""
    user_reported: bool = False
    verified: bool = False
    division: str = ""


@dataclass
class SensorData:
    id: int
    location: str
    lat: float
    lng: float
    water_level: float
    rainfall: float
    soil_moisture: float
    temperature: float
    humidity: float
    wind_speed: float
    timestamp: str
    division: str = ""


@dataclass
class UserReport:
    id: int
    user_name: str
    user_phone: str
    report_type: str  # 'flood', 'drought', 'waterlogging', 'infrastructure_damage'
    severity: str
    description: str
    location: str
    lat: float
    lng: float
    timestamp: str
    images: List[str] = None
    verified: bool = False
    upvotes: int = 0
    division: str = ""


@dataclass
class EmergencyService:
    id: int
    name: str
    type: str  # 'hospital', 'fire_station', 'police', 'rescue', 'shelter'
    phone: str
    address: str
    lat: float
    lng: float
    operational: bool = True
    capacity: int = 0
    current_load: int = 0


@dataclass
class WeatherStation:
    id: int
    name: str
    lat: float
    lng: float
    temperature: float
    humidity: float
    rainfall_24h: float
    wind_speed: float
    pressure: float
    visibility: float
    timestamp: str
    division: str = ""


@dataclass
class PredictionModel:
    model_type: str  # 'flood', 'drought', 'cyclone'
    accuracy: float
    prediction_24h: Dict
    prediction_7d: Dict
    last_updated: str


@dataclass
class FinanceData:
    title: str
    amount: float
    description: str
    type: str  # 'available' or 'allocated'
    category: str  # 'green_bonds', 'insurance', 'impact_investment'


@dataclass
class CommunityMetrics:
    communities_served: int
    trained_volunteers: int
    response_time_improvement: float
    property_damage_reduction: float


# Enhanced Data Store with Bangladesh-specific data
class DataStore:
    def __init__(self):
        self.alerts = []
        self.sensors = []
        self.user_reports = []
        self.emergency_services = []
        self.weather_stations = []
        self.predictions = {}
        self.finance_data = []
        self.community_metrics = CommunityMetrics(850, 12500, -45.0, 28500000.0)
        self.initialize_bangladesh_data()

    def initialize_bangladesh_data(self):
        # Initialize weather stations across Bangladesh
        for i, district in enumerate(BANGLADESH_DISTRICTS[:8]):
            station = WeatherStation(
                id=i + 1,
                name=f"{district['name']} Weather Station",
                lat=district['lat'] + random.uniform(-0.1, 0.1),
                lng=district['lng'] + random.uniform(-0.1, 0.1),
                temperature=random.uniform(25, 35),
                humidity=random.uniform(65, 90),
                rainfall_24h=random.uniform(0, 75),
                wind_speed=random.uniform(5, 25),
                pressure=random.uniform(1010, 1020),
                visibility=random.uniform(5, 15),
                timestamp=datetime.now().isoformat(),
                division=district['division']
            )
            self.weather_stations.append(station)

        # Initialize sensors across Bangladesh divisions
        for i, district in enumerate(BANGLADESH_DISTRICTS):
            sensor = SensorData(
                id=i + 1,
                location=district['name'],
                lat=district['lat'],
                lng=district['lng'],
                water_level=random.uniform(40, 85),
                rainfall=random.uniform(0, 60),
                soil_moisture=random.uniform(30, 75),
                temperature=random.uniform(25, 35),
                humidity=random.uniform(70, 95),
                wind_speed=random.uniform(5, 20),
                timestamp=datetime.now().isoformat(),
                division=district['division']
            )
            self.sensors.append(sensor)

        # Initialize emergency services
        emergency_services_data = [
            {'name': 'Dhaka Medical College Hospital', 'type': 'hospital', 'phone': '+880-2-8626812',
             'address': 'Ramna, Dhaka-1000', 'lat': 23.7268, 'lng': 90.3911, 'capacity': 2300},
            {'name': 'Chittagong Medical College Hospital', 'type': 'hospital', 'phone': '+880-31-2502203',
             'address': 'Panchlaish, Chittagong', 'lat': 22.3475, 'lng': 91.8123, 'capacity': 1500},
            {'name': 'Dhaka Fire Service', 'type': 'fire_station', 'phone': '999',
             'address': 'Tejgaon, Dhaka', 'lat': 23.7639, 'lng': 90.3889, 'capacity': 50},
            {'name': 'RAB Headquarters', 'type': 'police', 'phone': '999',
             'address': 'Uttara, Dhaka', 'lat': 23.8759, 'lng': 90.3795, 'capacity': 200},
            {'name': 'Coast Guard Bangladesh', 'type': 'rescue', 'phone': '+880-2-8316801',
             'address': 'Agargaon, Dhaka', 'lat': 23.7778, 'lng': 90.3647, 'capacity': 100},
            {'name': 'Dhaka Shelter Complex', 'type': 'shelter', 'phone': '+880-2-9558096',
             'address': 'Mirpur, Dhaka', 'lat': 23.8223, 'lng': 90.3654, 'capacity': 5000},
            {'name': 'Cox\'s Bazar Cyclone Shelter', 'type': 'shelter', 'phone': '+880-341-62324',
             'address': 'Cox\'s Bazar Sadar', 'lat': 21.4272, 'lng': 92.0058, 'capacity': 3000},
        ]

        for i, service in enumerate(emergency_services_data):
            emergency_service = EmergencyService(
                id=i + 1,
                name=service['name'],
                type=service['type'],
                phone=service['phone'],
                address=service['address'],
                lat=service['lat'],
                lng=service['lng'],
                operational=True,
                capacity=service['capacity'],
                current_load=random.randint(0, service['capacity'] // 3)
            )
            self.emergency_services.append(emergency_service)

        # Initialize sample user reports
        report_types = ['flood', 'waterlogging', 'infrastructure_damage', 'drought']
        for i in range(10):
            district = random.choice(BANGLADESH_DISTRICTS)
            report = UserReport(
                id=i + 1,
                user_name=f"নাগরিক {i + 1}",  # Citizen in Bengali
                user_phone=f"+880-1{random.randint(100000000, 999999999)}",
                report_type=random.choice(report_types),
                severity=random.choice(['Low', 'Medium', 'High']),
                description=self.get_sample_description(random.choice(report_types)),
                location=district['name'],
                lat=district['lat'] + random.uniform(-0.05, 0.05),
                lng=district['lng'] + random.uniform(-0.05, 0.05),
                timestamp=datetime.now().isoformat(),
                images=[],
                verified=random.choice([True, False]),
                upvotes=random.randint(0, 25),
                division=district['division']
            )
            self.user_reports.append(report)

        # Initialize finance data (Bangladesh-specific)
        self.finance_data = [
            FinanceData("বন্যা ত্রাণ তহবিল", 15000000, "Flood relief and rehabilitation fund", "available",
                        "emergency"),
            FinanceData("সাইক্লোন প্রস্তুতি", 8500000, "Cyclone preparedness infrastructure", "allocated",
                        "preparedness"),
            FinanceData("কমিউনিটি প্রশিক্ষণ", 3200000, "Community disaster preparedness training", "available",
                        "training"),
            FinanceData("আর্লি ওয়ার্নিং সিস্টেম", 12000000, "Early warning system enhancement", "allocated",
                        "technology"),
            FinanceData("গ্রিন ক্লাইমেট ফান্ড", 25000000, "Climate adaptation projects", "available", "green_bonds"),
            FinanceData("ইনফ্রাস্ট্রাকচার উন্নয়ন", 18000000, "Resilient infrastructure development", "allocated",
                        "infrastructure")
        ]

        # Initialize prediction models
        self.predictions = {
            'flood': PredictionModel(
                model_type='flood',
                accuracy=92.8,
                prediction_24h={'risk': 'Medium', 'percentage': 55, 'confidence': 0.89,
                                'affected_districts': ['Sylhet', 'Kurigram']},
                prediction_7d={'risk': 'High', 'percentage': 72, 'confidence': 0.85,
                               'affected_districts': ['Rangpur', 'Gaibandha', 'Lalmonirhat']},
                last_updated=datetime.now().isoformat()
            ),
            'cyclone': PredictionModel(
                model_type='cyclone',
                accuracy=94.5,
                prediction_24h={'risk': 'Low', 'percentage': 15, 'confidence': 0.91,
                                'landfall_probability': 'Bay of Bengal monitoring'},
                prediction_7d={'risk': 'Medium', 'percentage': 35, 'confidence': 0.87,
                               'coastal_districts': ['Cox\'s Bazar', 'Chittagong', 'Noakhali']},
                last_updated=datetime.now().isoformat()
            ),
            'drought': PredictionModel(
                model_type='drought',
                accuracy=88.9,
                prediction_24h={'risk': 'Low', 'percentage': 20, 'confidence': 0.83,
                                'affected_regions': ['Barind Tract']},
                prediction_7d={'risk': 'Medium', 'percentage': 45, 'confidence': 0.81,
                               'affected_regions': ['Northwestern Bangladesh']},
                last_updated=datetime.now().isoformat()
            )
        }

    def get_sample_description(self, report_type):
        descriptions = {
            'flood': "রাস্তায় পানি জমে গেছে। যানবাহন চলাচল বন্ধ।",  # Water logged on roads, transport stopped
            'waterlogging': "এলাকায় পানি নিষ্কাশন ব্যবস্থা কাজ করছে না।",  # Drainage system not working
            'infrastructure_damage': "সেতুর ক্ষতি হয়েছে। মেরামতের প্রয়োজন।",  # Bridge damaged, needs repair
            'drought': "পানির সংকট। টিউবওয়েল থেকে পানি আসছে না।"  # Water crisis, tube well not working
        }
        return descriptions.get(report_type, "পরিস্থিতি পর্যবেক্ষণ প্রয়োজন।")  # Situation needs monitoring


# Initialize data store
data_store = DataStore()


# Enhanced AI Prediction Engine for Bangladesh
class AIPredictionEngine:
    def __init__(self):
        self.models = {
            'flood_risk': self.flood_risk_model,
            'cyclone_risk': self.cyclone_risk_model,
            'drought_risk': self.drought_risk_model,
            'waterlogging_risk': self.waterlogging_risk_model
        }
        self.bangladesh_flood_factors = {
            'monsoon_season': 0.8,
            'river_proximity': 0.7,
            'elevation': 0.6,
            'urban_density': 0.5
        }

    def flood_risk_model(self, sensor_data: List[SensorData], location_lat: float = None,
                         location_lng: float = None) -> Dict:
        """Enhanced flood risk prediction for Bangladesh considering geographical factors"""
        avg_rainfall = np.mean([s.rainfall for s in sensor_data])
        avg_water_level = np.mean([s.water_level for s in sensor_data])
        avg_humidity = np.mean([s.humidity for s in sensor_data])

        # Bangladesh-specific risk calculation
        base_risk = (avg_rainfall * 0.4 + avg_water_level * 0.4 + avg_humidity * 0.2) / 100

        # Monsoon season factor (June-September)
        current_month = datetime.now().month
        if 6 <= current_month <= 9:
            base_risk *= 1.5  # Higher risk during monsoon

        # Location-based risk adjustment
        if location_lat and location_lng:
            # Check proximity to flood-prone areas
            for flood_area in FLOOD_PRONE_AREAS:
                distance = self.calculate_distance(location_lat, location_lng, flood_area['lat'], flood_area['lng'])
                if distance < 50:  # Within 50 km
                    risk_multiplier = {'Very High': 2.0, 'High': 1.5, 'Medium': 1.2}.get(flood_area['risk_level'], 1.0)
                    base_risk *= risk_multiplier
                    break

        risk_score = min(base_risk * 100, 100)

        if risk_score < 30:
            risk_level = 'Low'
        elif risk_score < 60:
            risk_level = 'Medium'
        elif risk_score < 85:
            risk_level = 'High'
        else:
            risk_level = 'Critical'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': {
                'rainfall': avg_rainfall,
                'water_level': avg_water_level,
                'humidity': avg_humidity,
                'monsoon_factor': 6 <= current_month <= 9
            },
            'confidence': random.uniform(0.85, 0.95),
            'recommendation': self.get_flood_recommendation(risk_level)
        }

    def cyclone_risk_model(self, weather_data: List[WeatherStation]) -> Dict:
        """Cyclone risk prediction for Bangladesh coastal areas"""
        avg_wind_speed = np.mean([w.wind_speed for w in weather_data])
        avg_pressure = np.mean([w.pressure for w in weather_data])
        coastal_humidity = np.mean([w.humidity for w in weather_data if
                                    w.name in ['Cox\'s Bazar Weather Station', 'Chittagong Weather Station']])

        # Cyclone formation indicators
        pressure_drop = 1013 - avg_pressure  # Standard pressure vs current
        wind_factor = avg_wind_speed / 50  # Normalize wind speed
        humidity_factor = coastal_humidity / 100

        risk_score = (pressure_drop * 0.5 + wind_factor * 30 + humidity_factor * 20)
        risk_score = min(max(risk_score, 0), 100)

        if risk_score < 25:
            risk_level = 'Low'
        elif risk_score < 50:
            risk_level = 'Medium'
        elif risk_score < 75:
            risk_level = 'High'
        else:
            risk_level = 'Critical'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': {
                'wind_speed': avg_wind_speed,
                'pressure_drop': pressure_drop,
                'coastal_humidity': coastal_humidity
            },
            'confidence': random.uniform(0.80, 0.92),
            'bay_of_bengal_status': 'Monitoring active',
            'recommendation': self.get_cyclone_recommendation(risk_level)
        }

    def drought_risk_model(self, sensor_data: List[SensorData]) -> Dict:
        """Drought risk prediction for Bangladesh"""
        avg_soil_moisture = np.mean([s.soil_moisture for s in sensor_data])
        avg_rainfall = np.mean([s.rainfall for s in sensor_data])
        avg_temperature = np.mean([s.temperature for s in sensor_data])

        # Drought indicators
        moisture_deficit = max(0, 60 - avg_soil_moisture) / 60
        rainfall_deficit = max(0, 30 - avg_rainfall) / 30
        heat_factor = max(0, avg_temperature - 30) / 10

        risk_score = (moisture_deficit * 40 + rainfall_deficit * 40 + heat_factor * 20) * 100
        risk_score = min(risk_score, 100)

        if risk_score < 30:
            risk_level = 'Low'
        elif risk_score < 60:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': {
                'soil_moisture': avg_soil_moisture,
                'rainfall_deficit': avg_rainfall,
                'temperature': avg_temperature
            },
            'confidence': random.uniform(0.78, 0.88),
            'recommendation': self.get_drought_recommendation(risk_level)
        }

    def waterlogging_risk_model(self, sensor_data: List[SensorData], user_reports: List[UserReport]) -> Dict:
        """Urban waterlogging risk for Bangladesh cities"""
        avg_rainfall = np.mean([s.rainfall for s in sensor_data])

        # Count recent waterlogging reports
        recent_reports = [r for r in user_reports if r.report_type == 'waterlogging'
                          and (datetime.now() - datetime.fromisoformat(r.timestamp)).hours < 24]
        report_factor = min(len(recent_reports) / 10, 1.0)  # Normalize to max 10 reports

        # Urban drainage capacity factor
        urban_factor = 0.7  # Assume limited drainage capacity in Bangladesh cities

        risk_score = (avg_rainfall * 0.6 + report_factor * 0.3 + urban_factor * 0.1) * 100
        risk_score = min(risk_score, 100)

        if risk_score < 40:
            risk_level = 'Low'
        elif risk_score < 70:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': {
                'current_rainfall': avg_rainfall,
                'user_reports': len(recent_reports),
                'drainage_capacity': urban_factor * 100
            },
            'confidence': random.uniform(0.75, 0.90),
            'recommendation': self.get_waterlogging_recommendation(risk_level)
        }

    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two coordinates in km"""
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
            math.radians(lat2)) * math.sin(dlng / 2) * math.sin(dlng / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def get_flood_recommendation(self, risk_level):
        recommendations = {
            'Low': "পরিস্থিতি পর্যবেক্ষণ করুন। জরুরি সামগ্রী প্রস্তুত রাখুন।",
            'Medium': "সতর্ক থাকুন। উঁচু স্থানে যাওয়ার পরিকল্পনা করুন।",
            'High': "উঁচু স্থানে সরে যান। জরুরি সেবায় যোগাযোগ করুন।",
            'Critical': "অবিলম্বে নিরাপদ স্থানে যান। ৯৯৯ নম্বরে কল করুন।"
        }
        return recommendations.get(risk_level, "পরিস্থিতি পর্যবেক্ষণ করুন।")

    def get_cyclone_recommendation(self, risk_level):
        recommendations = {
            'Low': "আবহাওয়া বুলেটিন অনুসরণ করুন।",
            'Medium': "জরুরি সামগ্রী প্রস্তুত করুন। সাইক্লোন শেল্টার চিহ্নিত করুন।",
            'High': "সাইক্লোন শেল্টারে যাওয়ার প্রস্তুতি নিন।",
            'Critical': "অবিলম্বে নিকটস্থ সাইক্লোন শেল্টারে যান।"
        }
        return recommendations.get(risk_level, "আবহাওয়া পরিস্থিতি পর্যবেক্ষণ করুন।")

    def get_drought_recommendation(self, risk_level):
        recommendations = {
            'Low': "পানি সাশ্রয় করুন।",
            'Medium': "পানির ব্যবহার কমান। বৃষ্টির পানি সংরক্ষণ করুন।",
            'High': "জরুরি পানির ব্যবস্থা করুন। কৃষি কার্যক্রম সামঞ্জস্য করুন।"
        }
        return recommendations.get(risk_level, "পানি সংরক্ষণ করুন।")

    def get_waterlogging_recommendation(self, risk_level):
        recommendations = {
            'Low': "নিষ্কাশন ব্যবস্থা পরিষ্কার রাখুন।",
            'Medium': "অপ্রয়োজনীয় যাতায়াত এড়িয়ে চলুন।",
            'High': "নিম্নাঞ্চল এড়িয়ে চলুন। বাড়িতে থাকুন।"
        }
        return recommendations.get(risk_level, "সতর্ক থাকুন।")


# Initialize AI engine
ai_engine = AIPredictionEngine()


# Background task for real-time data simulation
def update_sensor_data():
    """Background task to simulate real-time sensor data updates for Bangladesh"""
    while True:
        try:
            # Update sensor data
            for sensor in data_store.sensors:
                # Simulate realistic changes for Bangladesh weather patterns
                sensor.water_level = max(0, min(100, sensor.water_level + random.uniform(-3, 8)))
                sensor.rainfall = max(0, sensor.rainfall + random.uniform(-5, 15))
                sensor.soil_moisture = max(0, min(100, sensor.soil_moisture + random.uniform(-2, 4)))
                sensor.temperature = max(20, min(40, sensor.temperature + random.uniform(-1, 2)))
                sensor.humidity = max(60, min(100, sensor.humidity + random.uniform(-3, 3)))
                sensor.wind_speed = max(0, sensor.wind_speed + random.uniform(-2, 5))
                sensor.timestamp = datetime.now().isoformat()

            # Update weather stations
            for station in data_store.weather_stations:
                station.temperature = max(20, min(40, station.temperature + random.uniform(-1, 2)))
                station.humidity = max(60, min(100, station.humidity + random.uniform(-2, 3)))
                station.rainfall_24h = max(0, station.rainfall_24h + random.uniform(-5, 10))
                station.wind_speed = max(0, station.wind_speed + random.uniform(-3, 8))
                station.pressure = max(1000, min(1025, station.pressure + random.uniform(-2, 2)))
                station.visibility = max(1, min(20, station.visibility + random.uniform(-1, 2)))
                station.timestamp = datetime.now().isoformat()

            # Generate alerts based on conditions and user reports
            if random.random() < 0.15:  # 15% chance of generating alert
                alert_types = ['flood', 'cyclone', 'waterlogging', 'drought']
                alert_type = random.choice(alert_types)
                severity_levels = ['Low', 'Medium', 'High', 'Critical']
                severity = random.choice(severity_levels)

                # Choose a random district
                district = random.choice(BANGLADESH_DISTRICTS)

                alert = Alert(
                    id=int(time.time()),
                    type=alert_type,
                    severity=severity,
                    location=district['name'],
                    lat=district['lat'],
                    lng=district['lng'],
                    timestamp=datetime.now().isoformat(),
                    description=f"Auto-generated {alert_type} alert for {district['name']}",
                    user_reported=False,
                    verified=False,
                    division=district['division']
                )
                data_store.alerts.append(alert)
            time.sleep(10)
        except Exception as e:
            print(f"Error updating sensor data: {e}")
            time.sleep(5)


# --- GIS Real-Time Image and User Update Features ---

from flask import send_file
import folium
from io import BytesIO


# Endpoint: Generate GIS-based real-time map image with alerts, sensors, and user reports
@app.route('/gis_map')
def gis_map():
    # Center map on Bangladesh
    center_lat = (BANGLADESH_BOUNDS['north'] + BANGLADESH_BOUNDS['south']) / 2
    center_lng = (BANGLADESH_BOUNDS['east'] + BANGLADESH_BOUNDS['west']) / 2
    m = folium.Map(location=[center_lat, center_lng], zoom_start=7)

    # Add sensor markers
    for sensor in data_store.sensors:
        folium.CircleMarker(
            location=[sensor.lat, sensor.lng],
            radius=6,
            popup=f"Sensor: {sensor.location}<br>Water Level: {sensor.water_level:.1f}<br>Rainfall: {sensor.rainfall:.1f}",
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)

    # Add user report markers
    for report in data_store.user_reports:
        color = 'red' if report.report_type == 'flood' else 'orange'
        folium.Marker(
            location=[report.lat, report.lng],
            popup=f"User: {report.user_name}<br>Type: {report.report_type}<br>Severity: {report.severity}<br>Description: {report.description}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)

    # Add alert markers
    for alert in data_store.alerts[-20:]:
        folium.Marker(
            location=[alert.lat, alert.lng],
            popup=f"Alert: {alert.type}<br>Severity: {alert.severity}<br>Description: {alert.description}",
            icon=folium.Icon(color='purple', icon='exclamation-sign')
        ).add_to(m)

    # Save map to image
    img_data = m._to_png(5)
    return send_file(BytesIO(img_data), mimetype='image/png', as_attachment=False, download_name='gis_map.png')


# Endpoint: User can submit real-time report from their location (mobile/web)
@app.route('/submit_report', methods=['POST'])
def submit_report():
    data = request.json
    required_fields = ['user_name', 'user_phone', 'report_type', 'severity', 'description', 'lat', 'lng', 'location',
                       'division']
    if not all(field in data for field in required_fields):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    report = UserReport(
        id=len(data_store.user_reports) + 1,
        user_name=data['user_name'],
        user_phone=data['user_phone'],
        report_type=data['report_type'],
        severity=data['severity'],
        description=data['description'],
        location=data['location'],
        lat=float(data['lat']),
        lng=float(data['lng']),
        timestamp=datetime.now().isoformat(),
        images=data.get('images', []),
        verified=False,
        upvotes=0,
        division=data['division']
    )
    data_store.user_reports.append(report)
    return jsonify({'success': True, 'report_id': report.id})


# Endpoint: Get all user reports (optionally filter by location)
@app.route('/user_reports', methods=['GET'])
def get_user_reports():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    radius = request.args.get('radius', default=10, type=float)  # in km

    if lat is not None and lng is not None:
        # Filter reports within radius
        def within_radius(report):
            dlat = math.radians(report.lat - lat)
            dlng = math.radians(report.lng - lng)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(report.lat)) * math.sin(
                dlng / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = 6371 * c
            return distance <= radius

        filtered_reports = [asdict(r) for r in data_store.user_reports if within_radius(r)]
        return jsonify(filtered_reports)
    else:
        return jsonify([asdict(r) for r in data_store.user_reports])


# Endpoint: Real-time GIS dashboard (HTML page with live map and user report form)
@app.route('/gis_dashboard')
def gis_dashboard():
    return render_template('gis_dashboard.html')


@app.route('/')
def index():
    # Redirect to the GIS dashboard
    return render_template('gis_dashboard.html')


# --- HTML Template for GIS Dashboard (place this in templates/gis_dashboard.html) ---
'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AquaGuard GIS Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        #map { height: 500px; width: 100%; }
        #report-form { margin-top: 20px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/leaflet/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet/dist/leaflet.css"/>
</head>
<body>
    <h2>Bangladesh Disaster GIS Dashboard</h2>
    <div id="map"></div>
    <form id="report-form">
        <h3>Submit Real-Time Report</h3>
        <input type="text" id="user_name" placeholder="Your Name" required>
        <input type="text" id="user_phone" placeholder="Phone" required>
        <select id="report_type">
            <option value="flood">Flood</option>
            <option value="drought">Drought</option>
            <option value="waterlogging">Waterlogging</option>
            <option value="infrastructure_damage">Infrastructure Damage</option>
        </select>
        <select id="severity">
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
        </select>
        <input type="text" id="description" placeholder="Description" required>
        <input type="text" id="location" placeholder="Location (e.g., Dhaka)" required>
        <input type="text" id="division" placeholder="Division (e.g., Dhaka)" required>
        <input type="hidden" id="lat">
        <input type="hidden" id="lng">
        <button type="submit">Submit Report</button>
    </form>
    <script>
        let map = L.map('map').setView([23.6850, 90.3563], 7);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Load GIS map image as overlay (optional)
        // L.imageOverlay('/gis_map', [[26.6382, 88.0075], [20.3756, 92.6804]]).addTo(map);

        // Load user reports as markers
        function loadReports() {
            fetch('/user_reports')
                .then(res => res.json())
                .then(reports => {
                    reports.forEach(r => {
                        L.marker([r.lat, r.lng]).addTo(map)
                            .bindPopup(`<b>${r.user_name}</b><br>${r.report_type}<br>${r.severity}<br>${r.description}`);
                    });
                });
        }
        loadReports();

        // Get user's current position
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(pos) {
                document.getElementById('lat').value = pos.coords.latitude;
                document.getElementById('lng').value = pos.coords.longitude;
            });
        }

        // Handle report form submit
        document.getElementById('report-form').onsubmit = function(e) {
            e.preventDefault();
            let data = {
                user_name: document.getElementById('user_name').value,
                user_phone: document.getElementById('user_phone').value,
                report_type: document.getElementById('report_type').value,
                severity: document.getElementById('severity').value,
                description: document.getElementById('description').value,
                location: document.getElementById('location').value,
                division: document.getElementById('division').value,
                lat: document.getElementById('lat').value,
                lng: document.getElementById('lng').value
            };
            fetch('/submit_report', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            })
            .then(res => res.json())
            .then(resp => {
                if (resp.success) {
                    alert('Report submitted!');
                    loadReports();
                } else {
                    alert('Failed to submit report: ' + resp.error);
                }
            });
        };
    </script>
</body>
</html>
'''
# --- End of GIS and Local Involvement Update ---

if __name__ == "__main__":
    # Start background thread for sensor data updates
    sensor_thread = threading.Thread(target=update_sensor_data, daemon=True)
    sensor_thread.start()

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
