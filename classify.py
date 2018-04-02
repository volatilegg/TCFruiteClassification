# Import turicreate
import turicreate as tc

# Load images from FruitImages folder
data = tc.image_analysis.load_images("FruitImages", with_path=True)

# Define labels for classification
labels = ['Apple', 'Avocado', 'Banana', 'Plum', 'Strawberry']

def getLabel(path, labels=labels):
    for label in labels:
        if label in path:
            return label

# Define data label using in exported model is fruit
data['fruit'] = data['path'].apply(getLabel)

# Save examples into bananas-avocados.sframe for training model
data.save("fruits.sframe")

# Visualised labeled data
data.explore()

# Fetching all data from sframe file
allData = tc.SFrame("fruits.sframe")

# Spliting all labeled examples into 90% of training examples and 10% of testing examples
trainingData, testingData = allData.random_split(0.9)

# Define 2 types of architecture available to train data
squeezeNetModel = "squeezenet_v1.1"
resNetModel = "resnet-50"

# Create model with selected architecture
model = tc.image_classifier.create(trainingData, target="fruit", model=squeezeNetModel, max_iterations=100)

# Save predictions to mlmodel
predictions = model.classify(testingData)

# Evaluate created model and print out accuracy
evaluations = model.evaluate(testingData)
print "Accuracy : %s" % evaluations['accuracy']
print "Confusion Matrix : \n%s" % evaluations['confusion_matrix']

# Save model
model.save("fruits.model")

# Export model to CoreML model
model.export_coreml("FruitClassifierResNet.mlmodel")
