import os
import cv2
import numpy as np
import rospy
import tensorflow as tf

from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        # Initialize the light as unknown
        self.curr_light = TrafficLight.Unknown
        
        # Prepare the path to the traffic light simulator
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, "models/sim_model_mobilenet_v2_retrained.pb")
        rospy.loginfo("Traffic light classification model path set to: {}".format(model_path))
        
        self.tl_class_graph = tf.Graph
        with self.tl_class_graph.as_default():
            tl_class_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                tl_class_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(tl_class_graph_def, name='')
                
        # Set up the graph category indices
        self.category_index = {1: {'id': 1, 'name': 'Green'}, 
                               2: {'id': 2, 'name': 'Red'},
                               3: {'id': 3, 'name': 'Yellow'}, 
                               4: {'id': 4, 'name': 'off'}}
        
        # Set up config; End any operation past 10 seconds
        config = tf.ConfigProto()
        config.operation_timeout_in_ms = 10000 
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        # Finish setup and set the TF session
        self.sess = tf.Session(graph=self.tl_class_graph_def, config=config)
        
        # Define the tensors for detection_graph
        self.image_tensor = self.tl_class_graph_def.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.tl_class_graph_def.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.tl_class_graph_def.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.tl_class_graph_def.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.tl_class_graph_def.get_tensor_by_name('num_detections:0')  
        
        rospy.loginfo("TLClassifier initialization complete.")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Expand the image to run it through the model
        expanded_image = np.expand_dims(image, axis=0)

        # Retrieve the boxes, scores, classes, num from the tensors
        (boxes, scores, classes, num) = self.tf_session.run([self.detection_boxes, 
                                                             self.detection_scores, 
                                                             self.detection_classes, 
                                                             self.num_detections],
                                                             feed_dict={self.image_tensor: expanded_image})

        # Squeeze out the intended data
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # If the score for a light is not at least 0.5 then we will not consider it
        min_score_threshold = 0.5
        
        if scores is not None and scores[0] > min_score_threshold:
            class_name = self.category_index[classes[0]]['name']
            rospy.loginfo("Traffic light detected and classified as: {}".format(class_name))

            if classes[0] == 1: 
               return TrafficLight.GREEN
            elif classes[0] == 2: 
               return TrafficLight.RED
            elif classes[0] == 3: 
               return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
