import great_expectations as gx
import pandas as pd 

def validate_reviews_date(csv_path:str) -> bool:
    context = gx.get_context()
    df = pd.read_csv(csv_path)
    batch_data = context.sources.add_pandas("pandas_datasource").add_dataframe_asset("reviews_asset")
    batch = batch_data.get_batch(batch_parameters= {"dataframe": df})
    validator = context.get_validator(
        batch_data = batch,
        create_expectation_suite_with_name = "sentiment_validation_suite",  
    )

    validator.expect_column_to_exist("Review Text")
    validator.expect_column_to_exist("Rating")
    validator.expect_column_values_to_not_be_null("Rating")

    results = validator.validate()
    print("Data validation success: ", results['success'])
    return results['success']

if __name__ == "__main__":
    validate_reviews_date(r"/model_deployment/data/Womens Clothing E-Commerce Reviews.csv")