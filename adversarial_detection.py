# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model
from yolov3 import yolo_process_output, yolov3_tiny_anchors

class AdversarialDetection:
    def __init__(self, model, attack_type, monochrome, classes, xi=8/255.0, lr= 1 /255.0, fixed_area=None, max_iterations=None):
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        import keras.backend as K

        self.classes = classes
        self.num_classes = len(classes)
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()
        self.monochrome = monochrome

        if self.monochrome:
            self.noise = np.zeros((416, 416))
        else:
            # self.noise = np.random.uniform( -1.0, 1.0, size=(416, 416, 3))
            self.noise = np.zeros((416, 416, 3))

        self.xi = xi
        self.lr = lr
        self.adv_patch_boxes = []
        self.fixed = False
        
        self.iter = 0 # Attack iteration counter
        self.attack_active = False  # Track if the attack is active
        self.original_boxes_count = 0  # Track original boxes count
        self.current_boxes_count = 0  # Track current boxes
        self.percentage_increase = 0  # Track percentage increase in boxes

        self.fixed_area = fixed_area  # Predefined attack area [x, y, w, h]
        if self.fixed_area:
            # Convert to the format your code expects: [x, y, width, height]
            self.adv_patch_boxes = [self.fixed_area]
            print(f"Fixed attack area set to: {self.fixed_area}")
        self.max_iterations = max_iterations  # User-set iteration limit

        self.model = load_model(model)
        self.model.summary()
        self.attack_type = attack_type
        self.delta = None

        loss = 0
        for out in self.model.output:
            # Targeted One Box
            if attack_type == "one_targeted":
                loss = loss + K.max(K.sigmoid(K.reshape(out, (-1, 5 + self.num_classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.num_classes))[:, 5]))

            # Untargeted One Box
            # if attack_type == "one_untargeted":
            #     for i in range(0, self.classes):
            #         loss = loss + tf.reduce_sum(K.max(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, i+5])))

            # Targeted Multi boxes
            if attack_type == "multi_targeted":
                loss = loss + tf.reduce_sum(K.sigmoid(K.reshape(out, (-1, 5 + self.num_classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.num_classes))[:, 5]))

            # Untargeted Multi boxes
            if attack_type == "multi_untargeted":
                # loss = tf.reduce_sum(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 5:]))
                for i in range(0, self.num_classes):
                    loss = loss + tf.reduce_sum(K.sigmoid(K.reshape(out, (-1, 5 + self.num_classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.num_classes))[:, i+5]))

        # Reduce Random Noises
        # loss = loss - tf.reduce_sum(tf.image.total_variation(self.model.input))

        grads = K.gradients(loss, self.model.input)
        self.delta = K.sign(grads[0])
        self.sess = tf.compat.v1.keras.backend.get_session()

    def attack(self, input_cv_image):
        with self.graph.as_default():
            # Before attack, get original detection count
            if not self.attack_active:
                original_output = self.sess.run(self.model.output, 
                                            feed_dict={self.model.input: np.array([input_cv_image])})
                boxes, _, _ = yolo_process_output(original_output, yolov3_tiny_anchors, self.num_classes)
                self.original_boxes_count = len(boxes) if boxes is not None else 0

            # Draw each adversarial patch on the input image
            for box in self.adv_patch_boxes:
                # Apply same noise to all channels
                if self.monochrome:
                    input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] += self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                    input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] += self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                    input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] += self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                else:
                    input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] += self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]

                if(len(self.adv_patch_boxes) > 0 and (not self.fixed)):
                    # Check if we've reached max iterations
                    if self.max_iterations is not None and  self.iter >= self.max_iterations and not self.fixed:
                        self.fixed = True
                        self.attack_active = False
                    self.iter += 1
                    self.attack_active = True  # Mark the attack as active

                    grads = self.sess.run(self.delta, feed_dict={self.model.input:np.array([input_cv_image])})
                    if self.monochrome:
                        # For monochrome images, we average the gradients over RGB channels
                        # self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])] += self.lr / 3 * (grads[0, :, :, 0] + grads[0, :, :, 1] + grads[0, :, :, 2])[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]

                        # self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])] += self.lr * grads[0, :, :, 0][box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                        # self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])] += self.lr * grads[0, :, :, 1][box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                        self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])] += self.lr * grads[0, :, :, 2][box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                    else:
                        self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] += self.lr * grads[0, :, :, :][box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]
                    
                    # Clip the noise to ensure it stays within the bounds
                    self.noise = np.clip(self.noise, -self.xi, self.xi)
            
            # Ensure the input image is within the valid range
            input_cv_image = np.clip(input_cv_image, 0.0, 1.0)

            # Get adversarial output
            adv_output = self.sess.run(self.model.output, 
                                    feed_dict={self.model.input: np.array([input_cv_image])})
            boxes, _, _ = yolo_process_output(adv_output, yolov3_tiny_anchors, self.num_classes)
            self.current_boxes_count = len(boxes) if boxes is not None else 0
            
            # Calculate percentage increase
            self.percentage_increase = 0
            if self.original_boxes_count > 0:
                self.percentage_increase = ((self.current_boxes_count - self.original_boxes_count) / 
                                    self.original_boxes_count) * 100
                
            return input_cv_image, adv_output, {
                'attack_type': self.attack_type,
                'original_boxes': self.original_boxes_count,
                'current_boxes': self.current_boxes_count,
                'percentage_increase': self.percentage_increase,
                'iterations': self.iter
            }
        
    def collect_attack_metrics(self):
        # Collect metrics for analysis and visualization
        return {
            'attack_type': self.attack_type,
            'original_boxes': self.original_boxes_count,
            'current_boxes': self.current_boxes_count,
            'percentage_increase': self.percentage_increase,
            'iterations': self.iter,
            'noise_magnitude': np.mean(np.abs(self.noise)),
            'active_area': self.fixed_area if self.fixed_area else self.adv_patch_boxes[0] if self.adv_patch_boxes else None
        }