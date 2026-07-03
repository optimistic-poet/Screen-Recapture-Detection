from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# Change this if you want to test another image
image_path = "dataset/real/1000152297.jpg"

features = extractor.extract(image_path)

print("=" * 50)
print("Feature Extraction Successful!")
print("=" * 50)

print("Total Features:", len(features))
print("\nFirst 10 Features:")
print(features[:10])