from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
import os
import joblib
import pandas as pd
import json

from .models import Message, ChatMessage  # ✅ Forum + Chat models

# ===============================
# Model Loading
# ===============================

weather_model_path = os.path.join(settings.BASE_DIR, 'myapp', 'weather_model.pkl')
weather_model = joblib.load(weather_model_path)

crop_model_path = os.path.join(settings.BASE_DIR, 'myapp', 'crop_model.pkl')
label_encoder_path = os.path.join(settings.BASE_DIR, 'myapp', 'label_encoder.pkl')

if os.path.exists(crop_model_path) and os.path.exists(label_encoder_path):
    crop_model = joblib.load(crop_model_path)
    crop_label_encoder = joblib.load(label_encoder_path)
else:
    crop_model = None
    crop_label_encoder = None

planting_model_path = os.path.join(settings.BASE_DIR, "myapp", "weather_model.pkl")
planting_model = joblib.load(planting_model_path) if os.path.exists(planting_model_path) else None

WEATHER_CLASSES = {
    0: "rain 🌧️",
    1: "drizzle 🌦️",
    2: "snow ❄️",
    3: "sun ☀️",
    4: "fog 🌫️"
}

# === Load Yield Model & Metadata ===
MODEL_PATH = os.path.join(settings.BASE_DIR, "myapp", "yield_model.pkl")
META_PATH = os.path.join(settings.BASE_DIR, "myapp", "yield_metadata.pkl")

model = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH)

numeric_features = metadata.get("numeric_features", [])
categorical_features = metadata.get("categorical_features", [])
all_features = numeric_features + categorical_features

# Exclude unwanted columns
hidden_features = ["Unnamed: 0", "Year"]
visible_numeric = [f for f in numeric_features if f not in hidden_features]
visible_categorical = categorical_features.copy()

# ===============================
# Yield Prediction View
# ===============================

def yield_predict_view(request):
    prediction = None
    error_message = None
    user_input = {}

    if request.method == "POST":
        try:
            for col in visible_numeric:
                user_input[col] = float(request.POST.get(col))
            for col in visible_categorical:
                user_input[col] = request.POST.get(col)

            X_input = pd.DataFrame([[user_input.get(col, 0) for col in all_features]], columns=all_features)
            prediction = round(float(model.predict(X_input)[0]), 2)
        except Exception as e:
            error_message = str(e)

    context = {
        "numeric_features": visible_numeric,
        "categorical_features": visible_categorical,
        "prediction": prediction,
        "error_message": error_message,
    }
    return render(request, "myapp/yield_predict.html", context)

# ===============================
# Public Pages
# ===============================

def index(request):
    return render(request, 'myapp/index.html')

def about_us(request):
    return render(request, 'myapp/about_us.html')

# ===============================
# Protected Pages (login required)
# ===============================

@login_required(login_url='/accounts/login/')
def predict(request):
    return render(request, 'myapp/predict.html')

@login_required(login_url='/accounts/login/')
def blog_news(request):
    return render(request, 'myapp/blog_news.html')

@login_required(login_url='/accounts/login/')
def dashboard(request):
    return render(request, 'myapp/dashboard.html')

@login_required(login_url='/accounts/login/')
def help_support(request):
    return render(request, 'myapp/help_support.html')

@login_required(login_url='/accounts/login/')
def yield_predict(request):
    return render(request, 'myapp/yield_predict.html')

@login_required(login_url='/accounts/login/')
def adaptation_strategies(request):
    return render(request, 'myapp/adaptation_strategies.html')

@login_required(login_url='/accounts/login/')
def crop_rec(request):
    return render(request, 'myapp/crop_rec.html')

@login_required(login_url='/accounts/login/')
def community_forum(request):
    if request.method == "POST":
        text = request.POST.get("content")
        if text:
            Message.objects.create(
                user=request.user,
                content=text,
                is_admin_reply=request.user.is_staff
            )
        return redirect("community_forum")

    messages = Message.objects.order_by("created_at")
    return render(request, "myapp/community_forum.html", {"messages": messages})

# ===============================
# Authentication Views
# ===============================

from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.contrib import messages

def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Try another one.")
            return redirect("register")

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect("register")

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()

        login(request, user)

        # ✅ Send welcome email
        subject = "🌱 Welcome to AgriSmart!"
        context = {"username": username}
        message = render_to_string("emails/welcome_email.html", context)
        email_message = EmailMessage(subject, message, settings.DEFAULT_FROM_EMAIL, [email])
        email_message.content_subtype = "html"
        email_message.send()

        messages.success(request, f"Welcome {username}! You are now logged in.")
        return redirect("index")

    return render(request, "myapp/register.html")


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("index")
        else:
            return render(request, "myapp/accounts/login.html", {"error": "Invalid credentials."})

    return render(request, "myapp/accounts/login.html")


def logout_view(request):
    logout(request)
    return redirect("index")

# ===============================
# APIs (Weather, Crop, Planting)
# ===============================

@csrf_exempt
def predict_weather(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            X = pd.DataFrame([{
                'precipitation': float(data.get('precipitation', 0)),
                'temp_max': float(data.get('temp_max', 0)),
                'temp_min': float(data.get('temp_min', 0)),
                'wind': float(data.get('wind', 0))
            }])
            pred_index = int(weather_model.predict(X)[0])
            weather_type = WEATHER_CLASSES.get(pred_index, "Unknown")
            return JsonResponse({'prediction': weather_type})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def recommend_crop(request):
    if request.method == "POST":
        try:
            if crop_model is None or crop_label_encoder is None:
                return JsonResponse({'error': 'Crop model not available on server'}, status=500)

            data = json.loads(request.body)
            X = pd.DataFrame([{
                'Nitrogen': float(data.get('nitrogen', 0)),
                'Phosphorus': float(data.get('phosphorus', 0)),
                'Potassium': float(data.get('potassium', 0)),
                'Temperature': float(data.get('temperature', 0)),
                'Humidity': float(data.get('humidity', 0)),
                'pH_Value': float(data.get('ph', 0)),
                'Rainfall': float(data.get('rainfall', 0))
            }])

            pred_index = int(crop_model.predict(X)[0])
            crop_name = crop_label_encoder.inverse_transform([pred_index])[0]

            return JsonResponse({'recommended_crop': crop_name})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def predict_time(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            X = pd.DataFrame([{
                "precip_7d_mean": float(data.get("precip_7d_mean", 0)),
                "tempmax_7d_mean": float(data.get("tempmax_7d_mean", 0)),
                "tempmin_7d_mean": float(data.get("tempmin_7d_mean", 0)),
                "wind_7d_mean": float(data.get("wind_7d_mean", 0)),
                "precip_3d_sum": float(data.get("precip_3d_sum", 0)),
                "month": int(data.get("month", 1)),
                "day_of_year": int(data.get("day_of_year", 1)),
            }])

            if planting_model is None:
                return JsonResponse({"error": "Planting model not available"}, status=500)

            proba = planting_model.predict_proba(X)[0]
            pred = int(planting_model.predict(X)[0])

            recommendation = "Recommend planting" if pred == 1 else "Not suitable today"

            if pred == 0:
                if X["precip_3d_sum"].iloc[0] > 20:
                    recommendation = "Delay planting — waterlogging risk"
                elif X["tempmin_7d_mean"].iloc[0] < 4:
                    recommendation = "Delay planting — frost risk"
                elif X["tempmax_7d_mean"].iloc[0] > 30:
                    recommendation = "Consider irrigation or heat mitigation"
                else:
                    recommendation = "Monitor conditions — marginal suitability"

            return JsonResponse({
                "prediction": int(pred),
                "probability": float(max(proba)),
                "recommendation": recommendation
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=400)

# ===============================
# Chatbot API
# ===============================

@csrf_exempt
def receive_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message_text = data.get('message', '').strip()
            name = data.get('name', '')
            email = data.get('email', '')

            if not message_text:
                return JsonResponse({'status': 'error', 'message': 'Empty message'}, status=400)

            ChatMessage.objects.create(name=name, email=email, message=message_text)
            reply = "Thanks for your message! We'll get back to you soon. 🌿"
            return JsonResponse({'status': 'success', 'reply': reply})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)
