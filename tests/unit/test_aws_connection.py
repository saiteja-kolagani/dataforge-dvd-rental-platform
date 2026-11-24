import boto3

session = boto3.Session(
    profile_name="dataforge-dev",
    region_name="ap-south-1"
)

sts = session.client("sts")

identity = sts.get_caller_identity()

print("AWS Connection Successful!")
print("Account:", identity["Account"])
print("ARN:", identity["Arn"])