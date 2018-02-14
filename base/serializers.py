
from rest_framework.serializers import ModelSerializer
# from rest_framework.validators import (UniqueTogetherValidator,
#                                        UniqueValidator)

from .models import Mentor


class MentorSerializer(ModelSerializer):

    # def run_validators(self, value):
    #     """
    #     just for recursive link, if we wanna connect similarity in db
    #     :param value:
    #     :return:
    #     """
    #     for validator in self.validators:
    #         if (isinstance(validator, UniqueTogetherValidator) or
    #                 isinstance(validator, UniqueValidator)):
    #             self.validators.remove(validator)
    #     super(MentorSerializer, self).run_validators(value)

    def create(self, validated_data):
        instance, _ = Mentor.objects.get_or_create(**validated_data)
        return instance

    class Meta:
        model = Mentor
        fields = ('user_id', 'raw_data', 'latent_vector')

