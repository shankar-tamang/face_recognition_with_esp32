from django.shortcuts import render
from django.http import HttpResponse

def receive_data(request):

    print(request.POST)

    return HttpResponse("Message Received")
    