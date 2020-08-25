#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

class EventTimer
{
public:
	EventTimer()
	{
		cudaEventCreate(&m_start);
		cudaEventCreate(&m_stop);
	}

	~EventTimer()
	{
		cudaEventDestroy(m_start);
		cudaEventDestroy(m_stop);
	}

	void start(cudaStream_t stream = 0)
	{
		m_stream = stream;
		cudaEventRecord(m_start, stream);
		m_running = true;
	}

	void stop()
	{
		cudaEventRecord(m_stop, m_stream);
		m_running = false;
	}

	float elapsedMilliseconds()
	{
		if (m_running)
			cudaEventRecord(m_stop, m_stream);

		cudaEventSynchronize(m_stop);

		float elapsed_time;
		cudaEventElapsedTime(&elapsed_time, m_start, m_stop);

		return elapsed_time;
	}

	float elapsedSeconds()
	{
		return elapsedMilliseconds() / 1000.0f;
	}

private:
	cudaEvent_t m_start;
	cudaEvent_t m_stop;
	cudaStream_t m_stream = 0;
	bool m_running = false;
};

class Timer
{
public:
	void start()
	{
		m_StartTime = std::chrono::system_clock::now();
		m_bRunning = true;
	}

	void stop()
	{
		m_EndTime = std::chrono::system_clock::now();
		m_bRunning = false;
	}

	double elapsedMilliseconds()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime;

		if (m_bRunning)
		{
			endTime = std::chrono::system_clock::now();
		}
		else
		{
			endTime = m_EndTime;
		}

		return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
	}

	double elapsedSeconds()
	{
		return elapsedMilliseconds() / 1000.0;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;
	std::chrono::time_point<std::chrono::system_clock> m_EndTime;
	bool m_bRunning = false;
};

#endif // !TIMER_H
