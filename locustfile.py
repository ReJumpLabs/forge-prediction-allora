from locust import HttpUser, task, between

class APIUser(HttpUser):
    # Thời gian giữa các lần gửi yêu cầu của mỗi người dùng ảo (giây)
    wait_time = between(1, 1)

    # Định nghĩa một task để gửi yêu cầu GET
    @task
    def get_api(self):
        response = self.client.get("/inference/eth")