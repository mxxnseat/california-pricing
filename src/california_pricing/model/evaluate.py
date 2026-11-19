from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def evaluate(model, X, y, metrics=["rmse"]):
    result = {}
    prediction = model.predict(X)

    for metric in metrics:
        match metric:
            case "rmse":
                result[metric] = root_mean_squared_error(y, prediction)
            case "mae":
                result[metric] = mean_absolute_error(y, prediction)
            case "r2":
                result[metric] = r2_score(y, prediction)
    return result