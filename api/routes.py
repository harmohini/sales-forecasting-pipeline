"""
API Route Handlers.

Provides REST endpoints:
- GET  /health      : Service health check status
- GET  /model-info  : Saved model metadata and validation metrics
- POST /predict     : Real-time sales forecast prediction
"""

import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from api.schemas import ForecastRequest, ForecastResponse, HealthResponse, ModelInfoResponse
from src.forecasting.model_utils import load_model_metadata, load_model_artifact
from src.forecasting.predict import generate_forecast

logger = logging.getLogger("ingestion_logger")
router = APIRouter(tags=["forecasting"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns API status and service name."
)
def get_health():
    """
    Service health check endpoint.
    """
    return HealthResponse(status="ok", service="sales-forecasting-api")


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Information & Metrics",
    description="Returns metadata, training date ranges, feature count, and validation metrics for the trained model."
)
def get_model_info():
    """
    Return useful model metadata from models/model_metadata.json.
    """
    try:
        meta = load_model_metadata()
        m_metrics = meta.get("model_metrics", {})
        
        return ModelInfoResponse(
            model_type=meta.get("best_model", "HistGradientBoostingRegressor"),
            model_version=meta.get("model_version", "1.0.0"),
            validation_rmspe=m_metrics.get("rmspe", 0.0),
            validation_mae=m_metrics.get("mae", 0.0),
            validation_rmse=m_metrics.get("rmse", 0.0),
            training_data_range=f"{meta.get('training_data_start')} → {meta.get('training_data_end')}",
            validation_data_range=f"{meta.get('validation_data_start')} → {meta.get('validation_data_end')}",
            training_rows=meta.get("training_rows", 0),
            validation_rows=meta.get("validation_rows", 0),
            features_count=len(meta.get("features_used", []))
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata file not found. Please train the model first."
        )
    except Exception as e:
        logger.error(f"Error fetching model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model metadata."
        )


@router.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Generate Sales Forecast",
    description="Generates non-negative sales predictions using the Stage 3 trained model."
)
def predict_sales(request_data: ForecastRequest, request: Request):
    """
    Generate sales prediction for input features using singleton model artifact in app state.
    """
    # Retrieve cached singleton model artifact from FastAPI app.state
    artifact = getattr(request.app.state, "model_artifact", None)
    
    if artifact is None:
        # Fallback to loading model if app state not populated
        try:
            artifact = load_model_artifact()
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model artifact not available on server. Run training script first."
            )
        except Exception as e:
            logger.error(f"Error loading model artifact: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load model artifact."
            )

    try:
        # Convert Pydantic request model to single-row Pandas DataFrame
        input_dict = request_data.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Generate sales prediction using model pipeline
        predictions = generate_forecast(input_df, artifact=artifact)
        predicted_val = float(predictions[0])

        model_ver = artifact.get("model_version", "1.0.0")
        return ForecastResponse(
            predicted_sales=round(predicted_val, 2),
            model_version=model_ver
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating sales forecast."
        )
