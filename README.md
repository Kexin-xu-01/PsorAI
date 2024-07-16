# PsorAI
AI model to predict progression of psoriasis from skin images paired by KNN.

The model runs VGG19 on the psoriasis images to extract feature vectors from them. Then, it runs KNN on the feature vectors to put the most similar images into pairs. It then applies a random forest network to use the difference between the feature vectors in the image pair to predict if the pairs of images show an 'improvement', 'worsening', or 'no change' in psoriasis progression.
