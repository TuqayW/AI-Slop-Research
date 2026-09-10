from pathlib import Path

articles = {
"ai13_train_bridge_design.txt": """BRIDGES AND THE DISTRIBUTION OF WEIGHT.

The construction of a bridge requires more than sufficient strength in its
separate members. The weight placed upon the roadway must be conducted to the
supports in such a manner that no portion is required to perform a duty for
which it was not designed.

A beam resting upon two supports bends when loaded. The upper fibres are
compressed while the lower fibres are extended, and the amount of this change
depends upon the form and material of the beam. By arranging the members so
that the load follows several paths, the engineer may obtain greater strength
without a proportional increase in material.

The difficulty is increased when moving loads are considered. A bridge must
bear not only its own weight but also the varying pressure produced by carts,
engines, and other traffic. The structure therefore requires sufficient margin
to accommodate changes which cannot be predicted at one instant.

A good bridge is consequently not merely a strong object. It is an organized
system in which each member assists in carrying the load toward the supports.""",

"ai14_train_clock_mechanisms.txt": """THE ACCURACY OF CLOCK MECHANISMS.

The measurement of time depends upon the regularity of the mechanism employed.
A clock may contain a beautifully constructed train of wheels and yet fail to
keep satisfactory time if the regulating part is affected by temperature,
friction, or changes in the driving force.

The pendulum offers one means of maintaining regularity because its period is
comparatively stable. But even this arrangement is not entirely independent
of surrounding conditions. Changes in temperature alter the dimensions of the
rod, while variations in amplitude may influence the motion.

The object of the regulator is therefore to reduce these sources of error.
The driving force must be maintained within suitable limits, and the moving
parts must be constructed so that unnecessary friction is avoided.

The progress of clockmaking has consisted largely in reducing the number and
magnitude of such disturbances. Accuracy is obtained not by one ingenious
device alone, but by careful adjustment of all the conditions upon which the
motion depends.""",

"ai15_train_crop_rotation.txt": """THE ROTATION OF CROPS.

The productive value of a field cannot always be maintained by growing the
same crop year after year. Plants remove particular substances from the soil,
and repeated cultivation may therefore lead to a gradual decline in the
condition required for successful growth.

Rotation provides one means of avoiding this difficulty. A grain crop may be
followed by another plant having different requirements, and the interval may
also permit weeds and certain pests to be controlled. The improvement does not
come from the mere change of name of the crop. Its effect depends upon the
relations among the plants selected and the condition in which each leaves the
soil.

The farmer must consequently consider the succession as a whole. A rotation
which appears satisfactory on paper may fail if the climate, drainage, or
market conditions render one of its stages unsuitable.

The useful principle is that agriculture should be regarded as a continuing
system rather than as a series of isolated harvests.""",

"ai16_train_factory_belts.txt": """THE TRANSMISSION OF POWER BY BELTS.

In many workshops the motion of one shaft is communicated to another by means
of a belt. The arrangement is simple, but the regularity of the transmission
depends upon conditions which deserve attention.

A belt must possess sufficient tension to maintain contact with the pulleys.
If the tension is too small, slipping may occur; if it is too great, the load
upon the bearings is unnecessarily increased. The material of the belt and
the condition of the pulleys also influence the result.

As the distance between shafts increases, the arrangement becomes more
difficult to maintain. Changes in temperature and the gradual stretching of
the material alter the tension with time. Periodic adjustment is therefore
necessary.

The advantage of the belt system lies partly in its simplicity and its ability
to accommodate machinery arranged at convenient distances. Its successful
operation, however, depends upon maintaining a suitable relation among speed,
tension, friction, and load.""",

"ai17_train_metal_casting.txt": """THE COOLING OF CAST METALS.

When melted metal is poured into a mould, the solidification which follows is
not instantaneous. The outer surface may become solid while the interior
remains hot and fluid. The manner in which this cooling proceeds influences
the form and properties of the finished casting.

A mould which removes heat rapidly may produce a different structure from one
which cools the metal slowly. Unequal cooling can also produce internal
stresses because different portions of the casting contract at different
rates.

The thickness of the object is therefore an important consideration. A narrow
section may solidify while a thick portion remains fluid, and the resulting
contraction may leave a cavity or other defect.

Good casting practice requires attention to the shape of the mould as well as
to the composition of the metal. The finished object is determined not only by
what material is poured into the mould, but also by the history of its cooling.""",

"ai18_train_sewer_systems.txt": """THE CONSTRUCTION OF SEWERS.

A city sewer must carry away waste water with sufficient regularity to prevent
stagnation while avoiding unnecessary expense in construction. The inclination
of the pipe, its dimensions, and the quantity of water expected all enter into
the design.

A conduit which is too small may become obstructed, while one made excessively
large adds cost without providing a corresponding advantage. The velocity of
the current is also important, since a sluggish flow may permit material to
settle along the bottom.

Ventilation deserves attention because gases may accumulate when circulation
is imperfect. Access points are required so that obstructions can be discovered
and removed without disturbing the entire system.

The sewer is therefore a machine of a peculiar kind. It contains no engine,
yet its successful action depends upon the proper arrangement of passages,
gradients, inlets and outlets. A sound system is one whose separate parts work
together continuously under ordinary conditions.""",

"ai19_train_railway_brakes.txt": """THE OPERATION OF RAILWAY BRAKES.

The safety of a railway train depends upon the ability to reduce its motion
within a suitable distance. The greater the weight and speed of the train,
the more work must be performed by the braking apparatus before the wheels are
brought to rest.

A brake acts by pressing a frictional surface against a moving wheel or other
member. The resulting resistance converts mechanical energy into heat. If the
pressure is too small, the stopping distance becomes excessive; if it is
applied too suddenly, the wheels may cease to rotate while the train continues
to move, reducing the useful effect.

A successful system therefore requires control as well as strength. The
brakes on different vehicles must respond with sufficient uniformity that the
train does not undergo violent changes in longitudinal force.

The history of railway brakes shows again that an apparatus must be judged as
a system. The strength of a single lever tells little unless the complete
chain of action can be made reliable.""",

"ai20_train_photographic_exposure.txt": """THE DETERMINATION OF PHOTOGRAPHIC EXPOSURE.

The proper exposure of a photographic plate depends upon the quantity of light
received by the sensitive surface and upon the character of the subject. A
bright outdoor scene and a dim interior cannot be treated according to one
simple rule.

The opening of the lens controls the amount of light admitted, while the time
during which the plate is exposed determines how long the sensitive material
is affected. These quantities are related, and an alteration in one requires
a corresponding change in the other.

The subject itself introduces another consideration. A landscape containing
strong contrasts may demand a different treatment from an evenly illuminated
object. The purpose of the photograph also matters because detail in shadows
may sometimes be more important than the preservation of highlights.

Successful exposure is consequently a matter of judging several conditions
together rather than relying upon a single numerical table.""",

"ai21_train_lighthouse_lenses.txt": """THE LENSES OF LIGHTHOUSES.

The object of a lighthouse lens is to direct a considerable quantity of light
toward the horizon where it may be seen by distant vessels. A simple lamp
emits rays in many directions, much of whose illumination is consequently
wasted.

By arranging glass surfaces in suitable forms, the rays may be refracted into
a more useful path. The lens must be large enough to intercept the light
produced by the lamp and accurately shaped so that the desired beam is formed.

The revolving mechanism presents another problem. If the lens is caused to
move steadily, the observer at sea perceives a succession of flashes. The
interval and character of these flashes become part of the signal by which the
position of the lighthouse is recognized.

The apparatus therefore combines optics with mechanical motion. The brightness
of the lamp alone does not establish the usefulness of the lighthouse; the
light must be gathered, directed, and presented in a recognizable manner.""",

"ai22_valid_airship_navigation.txt": """NAVIGATION OF AIRSHIPS.

The movement of an airship introduces difficulties which are less conspicuous
in a vehicle supported by the ground. Wind affects the path directly, and a
change in altitude may alter both the speed and the character of the air
encountered.

The navigator must therefore distinguish between the direction in which the
ship points and the path actually followed over the earth. A steady wind from
one side may leave the vessel apparently well directed while carrying it far
from the intended course.

Instruments provide assistance, but measurements must be interpreted together.
A compass gives direction, while observation of landmarks or celestial bodies
may furnish an independent indication of progress.

The useful lesson is that navigation is not simply the steering of a machine.
It is the continuous comparison of intended motion with observed motion and
the correction of errors before they become large.""",

"ai23_valid_machinery_lubrication.txt": """THE USE OF LUBRICANTS IN MACHINERY.

When two solid surfaces move against each other, resistance arises from the
irregularities of the surfaces and from the deformation which accompanies
contact. A lubricant interposed between them may reduce this resistance and
carry away some of the heat produced by motion.

The mere presence of oil does not guarantee satisfactory operation. Its
viscosity must be suitable for the speed and pressure involved, and the supply
must reach the actual bearing surfaces. An excess may be as inconvenient as a
deficiency.

Dust and other foreign matter introduce another difficulty. If particles
enter the bearing, they may increase wear despite the presence of lubricant.
Cleanliness of the surrounding machinery is therefore part of the lubrication
problem.

The value of a lubricant is best judged by its effect upon the complete
machine: reduced friction, moderate temperature, and preservation of the
working surfaces over time.""",

"ai24_valid_factory_ventilation.txt": """VENTILATION OF FACTORIES.

A factory containing many workers must provide air in sufficient quantity to
replace that which has become warm or contaminated. Natural openings may be
adequate under favorable conditions, but their effect varies with the weather
and the arrangement of the building.

Mechanical ventilation offers greater control. Fans may create a definite
movement of air, but the success of the system depends upon the position of
the inlets and outlets. Air introduced at one point may pass directly to an
exit without properly reaching the occupied portions of the room.

Heat-producing machinery creates an additional demand because warm air tends
to rise and may collect beneath the roof. The arrangement should therefore
take account of the circulation produced by both machinery and occupants.

A satisfactory factory ventilation system is one that maintains acceptable
conditions throughout the working space, not merely one that moves a large
volume of air at the fan.""",

"ai25_valid_railway_timetables.txt": """THE PREPARATION OF RAILWAY TIMETABLES.

A railway timetable is a written representation of a complicated sequence of
movements. The arrival of one train may depend upon the departure of another,
while stations with limited track space may permit only certain combinations
of trains at one time.

The preparation of the schedule therefore requires attention to the physical
conditions of the line. The distance between stations, the speed of trains,
and the time required for loading and unloading all influence the possible
arrangements.

A timetable which appears convenient to passengers may be impracticable if it
allows no margin for ordinary delays. On the other hand, excessive intervals
waste the carrying capacity of the railway.

The best schedule is accordingly a compromise between speed, reliability and
efficient use of the line. Its success is measured not by the elegance of the
printed table but by the regularity with which the actual service can follow
it.""",

"ai26_test_electric_meters.txt": """THE MEASUREMENT OF ELECTRIC CURRENT.

The increasing use of electric power has created a need for instruments by
which the quantity of electricity delivered to an installation may be
determined. A mere indication that a circuit is carrying current does not
tell the engineer how much energy has been supplied over a period of time.

An electric meter must therefore respond to a quantity related to the current
and duration of the operation. The mechanism may produce motion proportional
to the electrical work performed, allowing the total consumption to be read
from a scale or register.

Accuracy depends upon more than the indicating pointer. Friction, temperature
and changes in the electrical conditions may introduce errors, and the meter
must be constructed so that such influences remain within suitable limits.

The value of the instrument lies in converting an invisible process into a
record which can be examined, compared and used for practical accounting.""",

"ai27_test_steam_boiler_inspection.txt": """INSPECTION OF STEAM BOILERS.

A steam boiler is a vessel in which pressure and temperature are maintained
for the production of useful power. Because the stresses imposed upon its
walls may be considerable, regular inspection is essential.

The visible condition of the metal provides only part of the information
required. Corrosion may progress where it cannot easily be seen, while scale
inside the boiler may interfere with the transfer of heat and alter the
conditions under which the vessel operates.

Inspection should consequently consider the construction, water condition,
working pressure and history of the apparatus. A record made on one occasion
is more useful when it can be compared with earlier observations.

The object is not merely to discover defects after they become serious, but to
recognize changes while there is still time to correct them. Reliable operation
depends upon continued examination as well as sound original construction.""",

"ai28_test_river_navigation.txt": """NAVIGATION OF RIVERS.

The navigation of a river depends upon more than the existence of a continuous
channel. Changes in depth, currents, bends and deposits of material may alter
the conditions under which vessels can pass.

A channel suitable in one season may become difficult in another if floods
remove material from one bank and deposit it elsewhere. The navigator must
therefore possess information concerning both the usual course of the river
and temporary changes.

Markers and charts provide assistance, while regular soundings reveal whether
the depth remains sufficient for vessels of a given draught. The cost of such
work must be compared with the value of maintaining reliable navigation.

A successful improvement does not attempt to force every river into the same
form. It begins with an understanding of the natural conditions and modifies
them only where the expected benefit justifies the work.""",

"ai29_test_municipal_gas_lighting.txt": """MUNICIPAL GAS LIGHTING.

The lighting of public streets requires a dependable supply of gas, suitable
lamps, and an arrangement that permits the flame to be maintained at useful
brightness. The expense is influenced not only by the quantity of gas consumed
but by losses occurring throughout the distribution system.

Pressure must be sufficient to carry the gas through the mains, yet excessive
pressure may create unnecessary difficulties at the burners. The condition of
the pipes also matters, since leakage represents both waste and reduced
service.

Street lamps require regular inspection because dirt, damaged fittings, or
irregular burners can reduce the useful light. The public sees only the flame,
but its quality depends upon the entire supply system behind it.

The economical provision of illumination therefore requires attention to
production, distribution, maintenance, and actual performance in the street.""",

"ai30_test_mining_drainage.txt": """DRAINAGE OF MINES.

Water entering a mine may come from rain, underground springs, or surrounding
strata. As workings are extended, the quantity may increase and interfere with
both safety and productive work.

Pumps must be selected according to the expected inflow and the height through
which the water must be raised. A machine sufficient under ordinary conditions
may prove inadequate during an unusual increase, so some margin of capacity is
desirable.

The arrangement of drains and sumps is equally important. Water should be
directed toward collecting points from which it can be removed without
hindering the workmen.

Mining drainage illustrates the importance of preparing for variations in
conditions. The pumping engine is only one part of the system by which water
is collected, controlled, and discharged."""
}

out = Path("dataset/ai")
out.mkdir(parents=True, exist_ok=True)

for name, text in articles.items():
    (out / name).write_text(text.strip() + "\n", encoding="utf-8")

print("Created", len(articles), "additional AI articles.")