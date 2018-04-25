import socket
import zmq
import node_pb2
import robot_behavior_pb2
import sys
import json

def getNodeParameters(name,parameterServerAddress,timeout):
    try:
        # using with keyword save the life! If not used, the unmanaged resources are getting cleaned up
        # and this caused problems with interacting with Naoqi proxy
        with zmq.Context.instance() as ctx: 
            with ctx.socket(zmq.REQ) as sock:
                #socket = context.socket(zmq.REQ)
                sock.connect(parameterServerAddress)
                print "Sending parameter retrieval request"
                sock.send(name)
                print "waiting parameter response"
                str = sock.recv()
                node = node_pb2.Node()
                print "parsing parameter response"
                node.ParseFromString(str)
                sock.close()
                #print node
                return node
    except:
        print "getNodeParameters : ", sys.exc_info()


def getParam(node,key,defaultValue):
    ret = defaultValue
    if node != None:
        for p in node.param:
            if p.key == key:
                ret = p.value.encode('utf-8') # needed to encode in utf-8 because naoqi doesn't accept unicode
    return ret

def getPublisherInfo(node,publisher_msg):
    ret = None
    if node != None:
        for p in node.publisher:
            if p.msg_type == publisher_msg:
                ret = p
                break
    return ret

def getSubscriberInfo(node,subscriber_msg):
    ret = None
    if node != None:
        for s in node.subscriber:
            if s.msg_type == subscriber_msg:
                ret = s
                break
    return ret

def get_behavior_modules(req,ip,port,timeout):
    try:
        # using with keyword save the life! If not used, the unmanaged resources are getting cleaned up
        # and this caused problems with interacting with Naoqi proxy
        with zmq.Context.instance() as ctx: 
            with ctx.socket(zmq.REQ) as sock:
                #socket = context.socket(zmq.REQ)
                sock.connect("%s:%s" % (ip,port))
                print "Sending behavior modules request"
                sock.send(req)
                print "Waiting behavior modules response"
                #  Get the reply.
                str = sock.recv(1024)
                print "Received behavior modules response"
                #print str
                result = json.loads(str)
                return result
    except:
        print "get_behavior_modules : ", sys.exc_info()

    return None

def register_behaviors(node,robot,parameterServerAddress,behaviors):
    print "Creating behavior module message"
    behaviorModule = robot_behavior_pb2.RobotBehaviorModule()
    behaviorModule.name = node.name
    behaviorModule.robot = robot

    port = int(getParam(node,"RequestServerPort", "5590"))
    ip = getParam(node,"RequestClientIP", "*")

    behaviorModule.responder.Host = ip
    behaviorModule.responder.Port = port

    for behavior in behaviors:
        print "Creating behavior description message", behavior
        desc = behaviorModule.behaviors.add()
        desc.name = behavior
        desc.function_name = behaviors[behavior]['function']
        args = behaviors[behavior]['args']
        for arg in args:
            desc_arg = desc.arg.add()
            desc_arg.name = arg
            desc_arg.place_holder = args[arg]['place_holder']
            desc_arg.value = str(args[arg]['value'])
            desc_arg.type = args[arg]['type']
        desc.type = robot_behavior_pb2.BehaviorDescription.Blocking
        desc.state = robot_behavior_pb2.BehaviorDescription.Idle

    #print behaviorModule

    try:
        with zmq.Context.instance() as ctx: 
            with ctx.socket(zmq.REQ) as sock:
                parts = []
                module_str = behaviorModule.SerializeToString();
                sock.connect(parameterServerAddress)
                parts.append("register_behaviors")
                parts.append(node.name.encode('utf-8'))
                parts.append(module_str)
                print "Sending register_motions message", parts
                sock.send_multipart(parts)
                response = sock.recv()
                sock.close()
                print response
    except:
        print "register_behaviors : ", sys.exc_info()
