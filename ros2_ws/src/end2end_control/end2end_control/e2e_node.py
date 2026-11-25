#!/usr/bin/env python3

import time

import cv2
import numpy as np
import onnxruntime
import rclpy

from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


PROVIDERS = ["CPUExecutionProvider"]


class EndToEndControlNode(Node):

    def __init__(self):

        super().__init__("e2e_control_node")

        # Parametros originales
        self.declare_parameter("model_path", "")
        self.declare_parameter(
            "image_topic",
            "/f1/camera/image_raw"
        )
        self.declare_parameter(
            "cmd_vel_topic",
            "/f1/cmd_vel"
        )

        # Parametros adicionales
        self.declare_parameter("max_speed", 7.0)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("max_angular_speed", 2.5)
        self.declare_parameter("watchdog_timeout", 0.5)
        self.declare_parameter("log_every", 30)

        model_path = self.get_parameter("model_path").value
        image_topic = self.get_parameter("image_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.max_speed = float(
            self.get_parameter("max_speed").value
        )

        self.speed_scale = float(
            self.get_parameter("speed_scale").value
        )

        self.max_angular_speed = float(
            self.get_parameter("max_angular_speed").value
        )

        self.watchdog_timeout = float(
            self.get_parameter("watchdog_timeout").value
        )

        self.log_every = max(
            1,
            int(self.get_parameter("log_every").value)
        )

        if not 0.0 < self.speed_scale <= 1.0:
            raise ValueError("speed_scale debe estar entre 0 y 1")

        if self.max_speed <= 0 or self.max_angular_speed <= 0:
            raise ValueError("Limites de velocidad invalidos")

        if self.watchdog_timeout <= 0:
            raise ValueError("watchdog_timeout invalido")

        # Estadisticas
        self.frame_count = 0
        self.total_inference_ms = 0.0
        self.total_callback_ms = 0.0

        self.last_image_time = None
        self.watchdog_active = False

        # Cargar modelo
        self.session = self.load_model_session(model_path)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # ROS 2
        self.cmd_pub = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.watchdog_timer = self.create_timer(
            0.1,
            self.watchdog_callback
        )

        self.get_logger().info(
            f"Subscribed to {image_topic}"
        )

        self.get_logger().info(
            f"Publishing on {cmd_vel_topic}"
        )

        self.get_logger().info(
            f"Max speed: {self.max_speed} m/s"
        )

        self.get_logger().info(
            f"Speed scale: {self.speed_scale}"
        )

        self.get_logger().info(
            "Autonomous controller ready"
        )

    def load_model_session(self, model_path):

        if not model_path:
            raise ValueError(
                "Parameter model_path is empty"
            )

        options = onnxruntime.SessionOptions()

        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1

        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = onnxruntime.InferenceSession(
            model_path,
            sess_options=options,
            providers=PROVIDERS
        )

        self.get_logger().info(
            f"Loaded model: {model_path}"
        )

        self.get_logger().info(
            f"Providers: {session.get_providers()}"
        )

        return session

    def publish_cmd(self, v, w):

        msg = Twist()

        msg.linear.x = float(v)
        msg.angular.z = float(w)

        self.cmd_pub.publish(msg)

    def image_to_numpy(self, msg):

        if msg.encoding.lower() not in ("rgb8", "bgr8"):
            raise ValueError(
                f"Unsupported encoding: {msg.encoding}"
            )

        if (msg.height, msg.width) != (480, 640):
            raise ValueError(
                f"Unexpected image size: "
                f"{msg.width}x{msg.height}"
            )

        # Considerar el padding de cada fila
        if msg.step < msg.width * 3:
            raise ValueError("Invalid image step")

        raw = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        rows = raw.reshape(
            msg.height,
            msg.step
        )

        image = rows[:, :msg.width * 3].reshape(
            msg.height,
            msg.width,
            3
        )

        if msg.encoding.lower() == "bgr8":

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        return image

    def preprocess_image(self, image_rgb):

        # Mismo recorte utilizado en entrenamiento
        cropped = image_rgb[200:, :, :]

        resized = cv2.resize(
            cropped,
            (200, 80),
            interpolation=cv2.INTER_AREA
        )

        normalized = (
            resized.astype(np.float32) / 255.0
        )

        # HWC -> CHW
        tensor = np.transpose(
            normalized,
            (2, 0, 1)
        )

        # CHW -> NCHW
        tensor = np.expand_dims(
            tensor,
            axis=0
        )

        return np.ascontiguousarray(
            tensor,
            dtype=np.float32
        )

    def image_callback(self, msg):

        callback_start = time.perf_counter()

        try:

            # 1. Convertir imagen ROS 2 a RGB
            image_rgb = self.image_to_numpy(msg)

            # 2. Preprocesamiento
            tensor = self.preprocess_image(image_rgb)

            # 3. Inferencia ONNX
            inference_start = time.perf_counter()

            prediction = self.session.run(
                [self.output_name],
                {self.input_name: tensor}
            )[0]

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            if prediction.shape != (1, 2):
                raise ValueError("Invalid model output shape")

            if not np.isfinite(prediction).all():
                raise ValueError("Invalid model prediction")

            v_norm = float(prediction[0, 0])
            w_norm = float(prediction[0, 1])

            # 4. Desnormalizacion
            v_raw = v_norm * 3.0 + 4.0
            w_raw = w_norm * 2.5

            # 5. Limites del modelo
            v_model = float(
                np.clip(v_raw, 4.0, 7.0)
            )

            w_model = float(
                np.clip(w_raw, -2.5, 2.5)
            )

            # 6. Escalado para pruebas
            v = min(
                v_model * self.speed_scale,
                self.max_speed
            )

            # Mantener aproximadamente la curvatura
            ratio = v / v_model

            w = float(
                np.clip(
                    w_model * ratio,
                    -self.max_angular_speed,
                    self.max_angular_speed
                )
            )

            # 7. Publicar velocidades
            self.publish_cmd(v, w)

            self.last_image_time = time.monotonic()

            if self.watchdog_active:

                self.watchdog_active = False

                self.get_logger().info(
                    "Camera connection recovered"
                )

            # 8. Estadisticas
            callback_ms = (
                time.perf_counter() - callback_start
            ) * 1000.0

            self.frame_count += 1

            self.total_inference_ms += inference_ms
            self.total_callback_ms += callback_ms

            if self.frame_count % self.log_every == 0:

                avg_inference = (
                    self.total_inference_ms /
                    self.frame_count
                )

                avg_callback = (
                    self.total_callback_ms /
                    self.frame_count
                )

                self.get_logger().info(
                    f"Frame={self.frame_count} | "
                    f"v={v:.3f} m/s | "
                    f"w={w:.3f} rad/s | "
                    f"Inference={avg_inference:.3f} ms | "
                    f"Callback={avg_callback:.3f} ms"
                )

        except Exception as error:

            self.get_logger().error(
                f"Image callback failed: {error}"
            )

            self.publish_cmd(0.0, 0.0)

    def watchdog_callback(self):

        if self.last_image_time is None:

            self.publish_cmd(0.0, 0.0)
            return

        elapsed = (
            time.monotonic() - self.last_image_time
        )

        if elapsed > self.watchdog_timeout:

            self.publish_cmd(0.0, 0.0)

            if not self.watchdog_active:

                self.watchdog_active = True

                self.get_logger().warning(
                    "Watchdog: camera timeout. Vehicle stopped."
                )

    def stop_vehicle(self):

        self.publish_cmd(0.0, 0.0)

        self.get_logger().info(
            "Vehicle stopped"
        )


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:

        node = EndToEndControlNode()

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        if node is not None:

            if rclpy.ok():

                node.stop_vehicle()

            node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":
    main()
