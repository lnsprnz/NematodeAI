from ultralytics.models.sam import SAM2VideoPredictor

# Create SAM2VideoPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
predictor = SAM2VideoPredictor(overrides=overrides)

video = 'NematodeAI/Da/C0098_cropped.mp4'

# Run inference with single point
# Read the CSV file
points_df = pd.read_csv('C:/Users/linus/NematodeAI/NematodeAI/Data/C0105.MP4_processedFrames/C0105_cropped_clicked_points.csv')

# Filter points for class 1 and 2 (exclude class 0/background)
valid_points = points_df[points_df['class'].isin([1, 2])]

# Convert to format needed by predictor
points = valid_points[['x', 'y']].values.tolist()
labels = valid_points['class'].map({1: 1, 2: 1}).tolist()  # Map both classes to 1 for segmentation

# Run inference with points from CSV
results = predictor(source=video, points=points, labels=labels)

# Run inference with multiple points
# results = predictor(source=video, points=[[920, 470], [909, 138]], labels=[1, 1])

# Run inference with multiple points prompt per object
# results = predictor(source=video, points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

# Run inference with negative points prompt
# results = predictor(source=video, points=[[[920, 470], [909, 138]]], labels=[[1, 0]])

