from django.conf.urls import url, include
from rest_framework import routers

from .views import MentorViewSet
from .views import MentorSimilarity


router = routers.DefaultRouter()

router.register('mentor', MentorViewSet)

urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^mentor/similarity', MentorSimilarity.as_view()),
]
