<template>
  <Page class="page">
    <ActionBar title="Pothole Detector" />
    <StackLayout>
      <Label text="Pothole Detection" class="title" />
      <Image ref="inputImage" src="~/assets/sample.jpg" width="300" height="300" />
      <Button text="Detect Potholes" @tap="detectPotholes" class="button" />
      <Image :src="outputImage" v-if="outputImage" width="300" height="300" />
      <Label :text="resultMessage" v-if="resultMessage" class="result" />
    </StackLayout>
  </Page>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';
import * as ort from 'onnxruntime-node';
import { knownFolders, path } from '@nativescript/core';

export default defineComponent({
  setup() {
    const outputImage = ref(null);
    const resultMessage = ref('');

    async function detectPotholes() {
      try {
        const documents = knownFolders.currentApp().path;
        const modelPath = path.join(documents, 'assets/yolo11n.onnx');
        const session = await ort.InferenceSession.create(modelPath);
        const imagePath = path.join(documents, 'assets/sample.jpg');

        // Load image and preprocess
        const inputTensor = preprocessImage(imagePath);
        const feeds = { images: inputTensor };
        const results = await session.run(feeds);
        const detections = results.output.data;

        // Process detections
        const labels = [];
        for (let i = 0; i < detections.length; i += 6) {
          const [x1, y1, x2, y2, conf, cls] = detections.slice(i, i + 6);
          if (conf > 0.5) {
            labels.push(`Class: ${cls}, Confidence: ${conf.toFixed(2)}`);
          }
        }

        if (labels.length > 0) {
          resultMessage.value = `Detected Potholes: \n${labels.join('\n')}`;
        } else {
          resultMessage.value = 'No potholes detected.';
        }

      } catch (error) {
        console.error('Error detecting potholes:', error);
        resultMessage.value = 'Error detecting potholes.';
      }
    }

    function preprocessImage(imagePath: string) {
      // Placeholder for image preprocessing
      // You can add actual image preprocessing here
      const tensor = new Float32Array(3 * 640 * 640).fill(0.5);  // Dummy data
      return new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
    }

    return { detectPotholes, outputImage, resultMessage };
  }
});
</script>

<style scoped>
.page {
  padding: 20px;
}
.title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 20px;
  text-align: center;
}
.button {
  margin: 20px 0;
  background-color: #007bff;
  color: #fff;
  padding: 10px 20px;
  border-radius: 8px;
}
.result {
  margin-top: 20px;
  color: #444;
  text-align: center;
}
</style>
