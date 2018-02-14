
from rest_framework.viewsets import ModelViewSet
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Mentor
from .serializers import MentorSerializer

# Create your views here.


class MentorViewSet(ModelViewSet):
    queryset = Mentor.objects.all()
    serializer_class = MentorSerializer


class MentorSimilarity(APIView):
    def post(self, request, format=None):
        """
        :param request: should contain raw_data parameter
        :param format:
        :return:
        """
        data = request.data
        if 'raw_data' not in data:
            return Response('user_id address should be in request',
                            status=status.HTTP_400_BAD_REQUEST)

        mentors = Mentor.objects.all().order_by('?')
        if mentors.count():
            return Response(data=mentors, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)
