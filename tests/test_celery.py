from test_tasks.tasks import add

if __name__ == "__main__":
    # Test the task
    result = add.delay(4, 4)
    print(f"Task ID: {result.id}")
    print(f"Result: {result.get(timeout=5)}")  # Wait up to 5 seconds for the result
