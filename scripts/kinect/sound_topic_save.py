"""
Module Docstring.

author:name
date:date
"""
import argparse
import logging
import nep
import json


# Initialize logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO
)  # Add filename='example.log' for logging to file


def argument_parser():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--node_name",
        type=str,
        default="kinect_sound",
        help="Name of the node: kinect_sound, kinect_human",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="data.txt",
        help="Name of the output file name.",
    )
    parser.add_argument(
        "--num_output",
        type=int,
        default=10,
        help="Number of items to save.",
    )
    return parser.parse_args()


def main(args):
    """
    Implement the main function.
    """
    node_name = args.node_name
    output_file_name = args.output_file_name

    # Create a new node
    node = nep.node(node_name)
    # Select the configuration of the subscriber
    conf = node.conf_sub()
    # Set the topic and the configuration of the subscriber
    sub = node.new_sub("/" + node_name, conf)

    data = {}
    data_defined = False

    # Read the information published in the topic registered
    i = 0
    while i < args.num_output:
        s, msg = sub.listen_info()
        if s:
            if data_defined == False:
                for key, value in msg.iteritems():
                    data[key] = []
                data_defined = True

            for key, value in msg.iteritems():
                data[key].append(value)

            log.info(msg["beam"])
            i = i + 1

    with open(output_file_name, "w") as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    args = argument_parser()
    main(args)
