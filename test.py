# -*- coding: utf-8 -*-
from socket import *
import sys

if len(sys.argv)<=1:
    print('Usage:"python ProxySserver.py sever_ip"\n[sever_ip: It is the IP Adress fo Proxy Server')
    sys.exit(2)

# Create a server socket, bind it to a port and start listening
tcpSerSock = socket(AF_INET,SOCK_STREAM)
tcpSerPort = 8888
tcpSerSock.bind (("", tcpSerPort))
tcpSerSock.listen(5)

while 1:
    print ('Ready ro serve...')
tcpCliSock, addr = tcpSerSock.accept()
print('Received a connection from :', addr)
message = tcpCliSock.recv(1024)
print(message)
Print(message.split()[1])
filename = message.split()[1].partition("/")[2]
print(filename)
fileExist = "false"
filetouse = "/" + filename
print(filetouse)
try:
    f = open(filetouse[1:], "r")
    outputdate = f.readlines()
    fileExist = "true"
    tcpCliSock.send("HTTP/1.0 200 OK\r\n")
    tcpCliSock.send("Content-Type:text/html\r\n")
    for i in range(0,len(outputdate)):
        tcpCliSock.send(outputdata[i])
    print('Read from cache')
except IOError:
    if fileExist == "false":
        c = socket(AF_INET, SOCK_STREAM)
hostn = filename.replace("www.", "", 1)
print(hostn)
try:
    c.connect(hostn, 80)
    fileobj = c.makefile('r', 0)
    fileobj.write("GET " + "http://" + filename + "http/1.0\n\n")
    tmpFile = open("./" + filename, "wb")
    for i in range(len(buff)):
        tmpFile.write(buff[i])
    tcpCliSock.send(buff[i])
except:
    print("Illegal request")
else:
    print("404 Error file not found. ")
tcpCliSock.close()

if __name__ == '__main__':
    main()

