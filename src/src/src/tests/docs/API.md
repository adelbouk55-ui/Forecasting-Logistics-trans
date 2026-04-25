# Forecasting-Logistics API Documentation

## Overview

The **Forecasting-Logistics-trans API** is a comprehensive REST API for time series forecasting and logistics demand prediction. It supports multiple forecasting models (Prophet, ARIMA, LSTM, XGBoost) and provides endpoints for data upload, model training, predictions, and performance metrics.

**Base URL:** `http://localhost:5000/api/v1`

**API Version:** `v1`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [Examples](#examples)
6. [Rate Limiting](#rate-limiting)
7. [Best Practices](#best-practices)
8. [Support](#support--troubleshooting)

---

## Authentication

Currently, the API is open (no authentication required). For production, implement JWT authentication:
