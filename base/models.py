from django.db import models
from jsonfield import JSONField

# Create your models here.


class Mentor(models.Model):

    raw_data = JSONField()
    user_id = models.IntegerField(primary_key=True)
    latent_vector = JSONField(default=None)


