import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent implements OnInit {
  linearModel: tf.Sequential;
  prediction: any;

  trainNewModel() {
    this.linearModel = tf.sequential();

    this.linearModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    this.linearModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    const xs = tf.tensor1d([3.2, 4.4, 5.5]);
    const ys = tf.tensor1d([1.6, 2.7, 3.5]);

    this.linearModel.fit(xs, ys).then(() => console.log('model trained'));
  }

  linearPrediction(val: any) {
    const output = this.linearModel.predict(
      tf.tensor2d([val], [1, 1]),
    ) as tf.Tensor<tf.Rank>;

    this.prediction = Array.from(output.dataSync())[0];
  }

  ngOnInit() {
    this.trainNewModel();
  }
}
