# For Program Synthesis
import openai
import programlib
from pexpect.exceptions import EOF, TIMEOUT

# Dealing with OpenAI API requires tenacity
from tenacity import retry, retry_if_exception_message, retry_if_exception_type, wait_random_exponential, stop_after_attempt

# For BOPTEST
import gymnasium as gym
from boptestGymEnv import BoptestGymEnv

import os

model = "gpt-4"
SYSTEM_MSG = os.environ.get('SYSTEM_MSG', 'You are a program synthesis system. Answer with code only.')
SUMMARY_FACTOR = int(os.environ.get('SUMMARY_FACTOR', 512))
N = int(os.environ.get('N', 4096))
ELECTRICITY_PRICE = os.environ.get('ELECTRICITY_PRICE', 'dynamic')
TIME_PERIOD = os.environ.get('TIME_PERIOD', 'peak_heat_day')
MAX_EPISODE_LENGTH = int(os.environ.get('MAX_EPISODE_LENGTH', 3600 * 24 * 7))

for config_var in ('SYSTEM_MSG', 'SUMMARY_FACTOR', 'N', 'ELECTRICITY_PRICE', 'TIME_PERIOD', 'MAX_EPISODE_LENGTH'):
  print(f'{config_var} = {eval(config_var)}')

# Make sure that the test case is bestest_hydronic_heat_pump
# TESTCASE=bestest_hydronic_heat_pump docker compose up
url = 'http://127.0.0.1:5000'

env = BoptestGymEnv(
  url                   = url,
  actions               = ['oveHeaPumY_u'],
  observations          = {'reaTZon_y':(280.,310.)}, 
  random_start_time     = True,
  max_episode_length    = MAX_EPISODE_LENGTH,
  warmup_period         = 3600,
  step_period           = 900,
  scenario              = {'electricity_price': ELECTRICITY_PRICE, 'time_period': TIME_PERIOD})

wrapped_env = gym.ObservationWrapper(env)

def brief():
  # bestest_hydronic_heat_pump test case,
  
  text = """
    Write a program to control a Hydronic Heat Pump in a simplified residential dwelling for a family of 5 members, modeled as a single thermal zone, located in Brussels, Belgium. The building envelope model is based on the BESTEST case 900 test case. but it is scaled to an area that is four times larger. The rectangular floor plan is 12 m by 16 m. Internal walls are configured such that there are around 12 rooms in the building. The builiding further contains 24 m2 of windows on the south facade.

    An air-to-water modulating heat pump of 15 kW nominal heating capacity extracts energy from the ambient air to heat up the floor heating emission system. A fan blows ambient air through the heat pump evaporator and circulation pump pumps water from the heat pump to the floorr when the heat pump is operating. 
    
    The program should be an infinite loop that reads zone operative temperature (Kelvin) with input() and outputs the heat pump modulating signal for compressor speed between 0 (not working) and 1 (working at maximum capacity) with print(). There should be no output other then the control signal. The program should be written in Python.
    """
  messages.append({"role": "user", "content": text})

def summarize(rollout):
  text = """
    Below you will find a rollout of the Hydronic Heat Pump environment.
    It represents a history of one thermostat control episode

    temp | reward | signal
    """
  
  for observation, reward, info, action in rollout:
    observation = "{:.3f}".format(observation[0])
    reward = "{:.3f}".format(reward) if reward else "None"
    action = "{:.3f}".format(action[0]) if action else "None"

    if action:
      text += observation + ' | ' + reward + ' | ' + action
    text += "\n"

  text += """
    Can you write a short summary of what happened?
    """
  
  messages = [
    {"role": "system", "content": 'You are a helpful assistant'},
    {"role": "user", "content": text}
  ]

  gpt(messages)
  return messages[-1]['content']

def debrief(rollout, total_reward):
  text = """
    Here's how it went:
    """
  start_idx = 0
  end_idx = SUMMARY_FACTOR

  while start_idx < len(rollout):
    text += summarize(rollout[start_idx:end_idx])
    start_idx += SUMMARY_FACTOR
    end_idx += SUMMARY_FACTOR
    text += "\n"

  text += f"""
    Total reward: {total_reward}

    Can you rewrite the program to achieve a higher reward?
    """
  messages.append({"role": "user", "content": text})

messages = [
    {"role": "system", "content": SYSTEM_MSG},
]

def prune_messages(*args):
  # message[0] is the system message
  print('Pruning the chat history to fit into context window')
  del messages[1]

@retry(retry=retry_if_exception_message(match=r'.*Please reduce the length.*'),
       after=prune_messages)
@retry(retry=retry_if_exception_type(openai.error.RateLimitError),
       wait=wait_random_exponential(),
       stop=stop_after_attempt(10))
def gpt(messages):
  completion = openai.ChatCompletion.create(
      model=model,
      messages=messages
    )
  messages.append(completion['choices'][0]['message'])

def extract_code(msg):
  if '```' in msg:
    msg = msg.split('```')[1]
    msg = msg.split('\n')[1:]
    return '\n'.join(msg)
  else:
    return msg

def test():
  code = extract_code(messages[-1]['content'])
  program = programlib.Program(code, language='Python')
  rollout = program.spawn().test(env)
  total_reward = program.avg_score
  return rollout, total_reward

brief()
print(messages[-1]['content'], flush=True)

for i in range(N):
  gpt(messages)
    
  print(messages[-1]['content'], flush=True)

  try:
    rollout, total_reward = test()
    debrief(rollout, total_reward)
  except ValueError as e:
    messages.append({"role": "user", "content": str(e)})
  except (OSError, EOF, TIMEOUT) as e:
    messages.append({"role": "user", "content": 'Your program doesn\'t seem to be expecting input.'})

  print(messages[-1]['content'], flush=True)