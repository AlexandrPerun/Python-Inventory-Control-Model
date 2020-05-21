import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from math import ceil
import os

# Generate universe variables
x_demand = np.arange(-50, 51, 1)
x_quantity_on_hand = np.arange(100, 201, 1)
x_inventory_action = np.arange(-50, 51, 1)

# Generate fuzzy membership functions
demand_fal_mf = fuzz.trapmf(x_demand, [-50, -50, -40, -20])
demand_de_mf = fuzz.trimf(x_demand, [-40, -20, 0])
demand_st_mf = fuzz.trimf(x_demand, [-20, 0, 20])
demand_inc_mf = fuzz.trimf(x_demand, [0, 20, 40])
demand_ris_mf = fuzz.trapmf(x_demand, [20, 40, 50, 50])

quantity_on_hand_min_mf = fuzz.trapmf(x_quantity_on_hand, [100, 100, 110, 130])
quantity_on_hand_low_mf = fuzz.trimf(x_quantity_on_hand, [110, 130, 150])
quantity_on_hand_ad_mf = fuzz.trimf(x_quantity_on_hand, [130, 150, 170])
quantity_on_hand_hi_mf = fuzz.trimf(x_quantity_on_hand, [150, 170, 190])
quantity_on_hand_ex_mf = fuzz.trapmf(x_quantity_on_hand, [170, 190, 200, 200])

inventory_action_nl_mf = fuzz.trapmf(x_inventory_action, [-50, -50, -45, -30])
inventory_action_nm_mf = fuzz.trimf(x_inventory_action, [-45, -30, -15])
inventory_action_ns_mf = fuzz.trimf(x_inventory_action, [-30, -15, 0])
inventory_action_o_mf = fuzz.trimf(x_inventory_action, [-15, 0, 15])
inventory_action_ps_mf = fuzz.trimf(x_inventory_action, [0, 15, 30])
inventory_action_pm_mf = fuzz.trimf(x_inventory_action, [15, 30, 45])
inventory_action_pl_mf = fuzz.trapmf(x_inventory_action, [30, 45, 50, 50])

# input and check of Demand and Quantity-On-Hand
flag = True
while flag:
    flag = False
    try:
        demand_input = int(input('Input Demand [-50, 50]: '))
        quantity_on_hand_input = int(input('Input Quantity-On-Hand [100, 200]: '))
        if (demand_input < -50 or demand_input > 50) or (quantity_on_hand_input < 100 or quantity_on_hand_input > 200):
            raise ValueError

    except ValueError:
        print('\nThe value entered must be an integer.\nThe Demand value must be in the range [-50, 50] and the Quantity-On-Hand must be in the range [100, 200].\n')
        flag = True

# find the strength of each membership function
demand_level_fal = fuzz.interp_membership(x_demand, demand_fal_mf, demand_input)
demand_level_de = fuzz.interp_membership(x_demand, demand_de_mf, demand_input)
demand_level_st = fuzz.interp_membership(x_demand, demand_st_mf, demand_input)
demand_level_inc = fuzz.interp_membership(x_demand, demand_inc_mf, demand_input)
demand_level_ris = fuzz.interp_membership(x_demand, demand_ris_mf, demand_input)

quantity_on_hand_level_min = fuzz.interp_membership(x_quantity_on_hand, quantity_on_hand_min_mf, quantity_on_hand_input)
quantity_on_hand_level_low = fuzz.interp_membership(x_quantity_on_hand, quantity_on_hand_low_mf, quantity_on_hand_input)
quantity_on_hand_level_ad = fuzz.interp_membership(x_quantity_on_hand, quantity_on_hand_ad_mf, quantity_on_hand_input)
quantity_on_hand_level_hi = fuzz.interp_membership(x_quantity_on_hand, quantity_on_hand_hi_mf, quantity_on_hand_input)
quantity_on_hand_level_ex = fuzz.interp_membership(x_quantity_on_hand, quantity_on_hand_ex_mf, quantity_on_hand_input)

# rules base and activation of our fuzzy membership functions at input values
active_rule1 = np.fmin(demand_level_fal, quantity_on_hand_level_ex)
inventory_action_activation_nl = np.fmin(active_rule1, inventory_action_nl_mf)

active_rule2 = np.fmax(np.fmin(demand_level_fal, quantity_on_hand_level_hi),
                       np.fmax(np.fmin(demand_level_de, quantity_on_hand_level_hi),
                               np.fmax(np.fmin(demand_level_de, quantity_on_hand_level_ex), np.fmin(demand_level_st, quantity_on_hand_level_ex))))
inventory_action_activation_nm = np.fmin(active_rule2, inventory_action_nm_mf)

active_rule3 = np.fmax(np.fmin(demand_level_fal, quantity_on_hand_level_ad),
                       np.fmax(np.fmin(demand_level_de, quantity_on_hand_level_ad), np.fmin(demand_level_st, quantity_on_hand_level_hi)))
inventory_action_activation_ns = np.fmin(active_rule3, inventory_action_ns_mf)

active_rule4 = np.fmax(np.fmin(demand_level_fal, quantity_on_hand_level_min),
                       np.fmax(np.fmin(demand_level_fal, quantity_on_hand_level_low),
                               np.fmax(np.fmin(demand_level_de, quantity_on_hand_level_low),
                                       np.fmax(np.fmin(demand_level_st, quantity_on_hand_level_ad),
                                               np.fmax(np.fmin(demand_level_inc, quantity_on_hand_level_hi),
                                                       np.fmax(np.fmin(demand_level_inc, quantity_on_hand_level_ex), np.fmin(demand_level_ris, quantity_on_hand_level_ex)))))))
inventory_action_activation_o = np.fmin(active_rule4, inventory_action_o_mf)

active_rule5 = np.fmax(np.fmin(demand_level_de, quantity_on_hand_level_min),
                       np.fmax(np.fmin(demand_level_st, quantity_on_hand_level_low),
                               np.fmax(np.fmin(demand_level_inc, quantity_on_hand_level_ad), np.fmin(demand_level_ris, quantity_on_hand_level_hi))))
inventory_action_activation_ps = np.fmin(active_rule5, inventory_action_ps_mf)

active_rule6 = np.fmax(np.fmin(demand_level_st, quantity_on_hand_level_min),
                       np.fmax(np.fmin(demand_level_inc, quantity_on_hand_level_min),
                               np.fmax(np.fmin(demand_level_inc, quantity_on_hand_level_low), np.fmin(demand_level_ris, quantity_on_hand_level_ad))))
inventory_action_activation_pm = np.fmin(active_rule6, inventory_action_pm_mf)

active_rule7 = np.fmax(np.fmin(demand_level_ris, quantity_on_hand_level_min), np.fmin(demand_level_ris, quantity_on_hand_level_low))
inventory_action_activation_pl = np.fmin(active_rule7, inventory_action_pl_mf)

inventory_action_0 = np.zeros_like(x_inventory_action)

# Aggregate all three output membership functions together
aggregated = np.fmax(inventory_action_activation_nl,
                     np.fmax(inventory_action_activation_nm,
                             np.fmax(inventory_action_activation_ns,
                                     np.fmax(inventory_action_activation_o,
                                             np.fmax(inventory_action_activation_ps,
                                                     np.fmax(inventory_action_activation_pm, inventory_action_activation_pl))))))

# Calculate defuzzified result
inventory_action = fuzz.defuzz(x_inventory_action, aggregated, 'mom')
inventory_action_activation = fuzz.interp_membership(x_inventory_action, aggregated, inventory_action)

# Calculate  the necessary inventory action
quantity_on_hand_must_be = quantity_on_hand_input * (1 + inventory_action/100)
quantity_on_hand_must_be = ceil(quantity_on_hand_must_be)
necessary_inventory_action = quantity_on_hand_must_be - quantity_on_hand_input
if necessary_inventory_action < 0:
    print('\nInventory action: {0} parts necessary to reduce\n'.format(necessary_inventory_action))
elif necessary_inventory_action > 0:
    print('\nInventory action: {0} parts to order\n'.format(necessary_inventory_action))
else:
    print('\nIn stock now the required number of parts. No inventory action needed\n')

def visualize_mf():
    # Visualize these universes and membership functions
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(x_demand, demand_fal_mf, 'red', linewidth=1.5, label='falling')
    ax0.plot(x_demand, demand_de_mf, 'orange', linewidth=1.5, label='decreased')
    ax0.plot(x_demand, demand_st_mf, 'yellow', linewidth=1.5, label='steady')
    ax0.plot(x_demand, demand_inc_mf, 'green', linewidth=1.5, label='increased')
    ax0.plot(x_demand, demand_ris_mf, 'c', linewidth=1.5, label='rising')
    ax0.set_title('Demand')
    ax0.legend()

    ax1.plot(x_quantity_on_hand, quantity_on_hand_min_mf, 'red', linewidth=1.5, label='minimal')
    ax1.plot(x_quantity_on_hand, quantity_on_hand_low_mf, 'orange', linewidth=1.5, label='low')
    ax1.plot(x_quantity_on_hand, quantity_on_hand_ad_mf, 'yellow', linewidth=1.5, label='adequate')
    ax1.plot(x_quantity_on_hand, quantity_on_hand_hi_mf, 'green', linewidth=1.5, label='high')
    ax1.plot(x_quantity_on_hand, quantity_on_hand_ex_mf, 'c', linewidth=1.5, label='excessive')
    ax1.set_title('Quantity-On-Hand')
    ax1.legend()

    ax2.plot(x_inventory_action, inventory_action_nl_mf, 'red', linewidth=1.5, label='negative_large')
    ax2.plot(x_inventory_action, inventory_action_nm_mf, 'orange', linewidth=1.5, label='negative_medium')
    ax2.plot(x_inventory_action, inventory_action_ns_mf, 'yellow', linewidth=1.5, label='negative_small')
    ax2.plot(x_inventory_action, inventory_action_o_mf, 'green', linewidth=1.5, label='zero')
    ax2.plot(x_inventory_action, inventory_action_ps_mf, 'c', linewidth=1.5, label='positive_small')
    ax2.plot(x_inventory_action, inventory_action_pm_mf, 'blue', linewidth=1.5, label='positive_medium')
    ax2.plot(x_inventory_action, inventory_action_pl_mf, 'm', linewidth=1.5, label='positive_large')
    ax2.set_title('Inventory Action')
    ax2.legend()

    # Turn off top/right axes
    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()

def visualize_membership_activity():
    # Visualize Output membership activity
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_inventory_action, inventory_action_0, inventory_action_activation_nl, facecolor='red', alpha=0.7)
    ax0.plot(x_inventory_action, inventory_action_nl_mf, 'red', linewidth=0.5, linestyle='--', )
    ax0.fill_between(x_inventory_action, inventory_action_0, inventory_action_activation_nm, facecolor='orange',
                     alpha=0.7)
    ax0.plot(x_inventory_action, inventory_action_nm_mf, 'orange', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_inventory_action, inventory_action_0, inventory_action_activation_ns, facecolor='yellow',
                     alpha=0.7)
    ax0.plot(x_inventory_action, inventory_action_ns_mf, 'yellow', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_inventory_action, inventory_action_0, inventory_action_activation_o, facecolor='green',
                     alpha=0.7)
    ax0.plot(x_inventory_action, inventory_action_o_mf, 'green', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_inventory_action, inventory_action_0, inventory_action_activation_ps, facecolor='c', alpha=0.7)
    ax0.plot(x_inventory_action, inventory_action_ps_mf, 'c', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_inventory_action, inventory_action_0, inventory_action_activation_pm, facecolor='blue',
                     alpha=0.7)
    ax0.plot(x_inventory_action, inventory_action_pm_mf, 'blue', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_inventory_action, inventory_action_0, inventory_action_activation_pl, facecolor='m', alpha=0.7)
    ax0.plot(x_inventory_action, inventory_action_pl_mf, 'm', linewidth=0.5, linestyle='--')
    ax0.set_title('Output membership activity')

    # Turn off top/right axes
    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()

def visualize_result():
    # Visualize Aggregated membership and result (line)
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_inventory_action, inventory_action_nl_mf, 'red', linewidth=0.5, linestyle='--')
    ax0.plot(x_inventory_action, inventory_action_nm_mf, 'orange', linewidth=0.5, linestyle='--')
    ax0.plot(x_inventory_action, inventory_action_ns_mf, 'yellow', linewidth=0.5, linestyle='--')
    ax0.plot(x_inventory_action, inventory_action_o_mf, 'green', linewidth=0.5, linestyle='--')
    ax0.plot(x_inventory_action, inventory_action_ps_mf, 'c', linewidth=0.5, linestyle='--')
    ax0.plot(x_inventory_action, inventory_action_pm_mf, 'blue', linewidth=0.5, linestyle='--')
    ax0.plot(x_inventory_action, inventory_action_pl_mf, 'm', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_inventory_action, inventory_action_0, aggregated, facecolor='gold', alpha=0.7)
    ax0.plot([inventory_action, inventory_action], [0, inventory_action_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Aggregated membership and result (line)')

    # Turn off top/right axes
    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()

answer = input('Display graphs of membership functions? (input "yes" or "no") ... ')
answer.lower()
if answer == 'yes':
    visualize_mf()

answer = input('Display activity graph of membership functions? (input "yes" or "no") ... ')
answer.lower()
if answer == 'yes':
    visualize_membership_activity()

answer = input('Display the resulting graph? (input "yes" or "no") ... ')
answer.lower()
if answer == 'yes':
    visualize_result()

os.system("pause")