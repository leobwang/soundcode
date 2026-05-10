#!/usr/bin/env python3
"""Set up a minimal EC2 bastion/jump host for SSH tunneling."""

import boto3
import time
import json

ec2 = boto3.client("ec2", region_name="us-east-1")
ec2_resource = boto3.resource("ec2", region_name="us-east-1")

INSTANCE_NAME = "leo-ssh-bastion"
KEY_NAME = "leo-bastion-key"
SG_NAME = "leo-bastion-sg"

# 1. Find or create security group allowing SSH from anywhere
vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
vpc_id = vpcs["Vpcs"][0]["VpcId"]
print(f"Using VPC: {vpc_id}")

try:
    sg = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="SSH bastion - allow inbound SSH",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH from anywhere"}],
            }
        ],
    )
    print(f"Created security group: {sg_id}")
except ec2.exceptions.ClientError as e:
    if "already exists" in str(e):
        sgs = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
        )
        sg_id = sgs["SecurityGroups"][0]["GroupId"]
        print(f"Security group already exists: {sg_id}")
    else:
        raise

# 2. Import this machine's SSH key
pub_key = open("/home/leobwang/.ssh/id_ed25519.pub", "rb").read()
try:
    ec2.import_key_pair(KeyName=KEY_NAME, PublicKeyMaterial=pub_key)
    print(f"Imported key pair: {KEY_NAME}")
except ec2.exceptions.ClientError as e:
    if "already exists" in str(e):
        print(f"Key pair already exists: {KEY_NAME}")
    else:
        raise

# 3. Find latest Amazon Linux 2023 AMI (free tier, small)
images = ec2.describe_images(
    Owners=["amazon"],
    Filters=[
        {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]},
        {"Name": "state", "Values": ["available"]},
        {"Name": "architecture", "Values": ["x86_64"]},
    ],
)
ami = sorted(images["Images"], key=lambda x: x["CreationDate"], reverse=True)[0]
ami_id = ami["ImageId"]
print(f"Using AMI: {ami_id} ({ami['Name']})")

# 4. Launch instance
instances = ec2_resource.create_instances(
    ImageId=ami_id,
    InstanceType="t2.micro",
    KeyName=KEY_NAME,
    MinCount=1,
    MaxCount=1,
    TagSpecifications=[
        {
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": INSTANCE_NAME}],
        }
    ],
    NetworkInterfaces=[
        {
            "DeviceIndex": 0,
            "AssociatePublicIpAddress": True,
            "Groups": [sg_id],
        }
    ],
)

instance = instances[0]
print(f"Launched instance: {instance.id}")
print("Waiting for instance to be running...")
instance.wait_until_running()
instance.reload()

public_ip = instance.public_ip_address
print(f"\n{'='*50}")
print(f"Bastion instance is ready!")
print(f"  Instance ID: {instance.id}")
print(f"  Public IP:   {public_ip}")
print(f"  User:        ec2-user")
print(f"  Key:         ~/.ssh/id_ed25519 (this machine's key)")
print(f"{'='*50}")

# Save info for later
info = {
    "instance_id": instance.id,
    "public_ip": public_ip,
    "security_group_id": sg_id,
    "region": "us-east-1",
}
with open("/home/leobwang/.ssh/bastion_info.json", "w") as f:
    json.dump(info, f, indent=2)
print(f"\nBastion info saved to ~/.ssh/bastion_info.json")
