# Generated by Django 2.0.2 on 2018-02-14 14:57

from django.db import migrations, models
import jsonfield.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Mentor',
            fields=[
                ('raw_data', jsonfield.fields.JSONField()),
                ('user_id', models.IntegerField(primary_key=True, serialize=False)),
                ('latent_vector', jsonfield.fields.JSONField()),
            ],
        ),
    ]
