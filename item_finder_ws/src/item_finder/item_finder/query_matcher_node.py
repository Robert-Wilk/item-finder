"""
query_matcher_node
-------------------
Subscribes to /find (free text, e.g. "tv remote"), resolves it to a
canonical COCO class via item_finder.target_classes, and publishes the
result to /target_class as a LATCHED message (transient local QoS) so
that anything subscribing later - e.g. search_state_node restarting,
or a debug tool - immediately gets the current target without waiting
for the next /find message.

Also publishes /find_result with a human-readable status, mainly so a
failed match is visible/debuggable instead of silently doing nothing.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import String

from item_finder.target_classes import resolve_target


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)


class QueryMatcherNode(Node):
    def __init__(self):
        super().__init__('query_matcher_node')

        self.target_pub = self.create_publisher(String, '/target_class', LATCHED_QOS)
        self.result_pub = self.create_publisher(String, '/find_result', 10)

        self.find_sub = self.create_subscription(
            String, '/find', self.on_find, 10
        )

        self.get_logger().info('query_matcher_node ready. Listening on /find.')

    def on_find(self, msg: String):
        query = msg.data
        resolved, via = resolve_target(query)

        if resolved is None:
            result = f'No match found for "{query}". Try one of the known objects.'
            self.get_logger().warn(result)
            self.result_pub.publish(String(data=result))
            return

        self.get_logger().info(f'"{query}" -> "{resolved}" (matched via {via})')
        self.target_pub.publish(String(data=resolved))
        self.result_pub.publish(String(data=f'Searching for: {resolved} (matched via {via})'))


def main(args=None):
    rclpy.init(args=args)
    node = QueryMatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
