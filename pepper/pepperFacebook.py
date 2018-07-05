from fbchat import Client
from fbchat.models import *

myUsername = "pattarsuraj@gmail.com"
myPassword = "TuIlvibgotVefM2"
client1 = Client(myUsername, myPassword)
#print('Own id: {}'.format(client.uid))
pepperUsername = "gentianeventurelab@gmail.com"
pepperPassword = "tokyo2009Noko"
client2 = Client(pepperUsername, pepperPassword)
#friend = client.searchForUsers(pepperUsername)
#friends = friend[0]
client2.send(Message(text='Hi Suraj!'), thread_id=client1.uid, thread_type=ThreadType.USER)
client1.send(Message(text='Hi Pepper!'), thread_id=client2.uid, thread_type=ThreadType.USER)
#if sent:
#    print("Message sent succesfully!")


